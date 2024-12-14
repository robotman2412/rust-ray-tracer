
use std::{borrow::BorrowMut, sync::{Arc, Mutex}, thread::{spawn, JoinHandle}};

use rand::rngs::ThreadRng;
use sdl2::{pixels::Color, rect::Point, render::{Canvas, RenderTarget}};

use crate::*;
use crate::matrix::*;
use crate::scene::*;

pub struct Tracer {
    pub max_reflect:     u16,
    pub max_refract:     u16,
    pub fov:             f64,
    pub reflect_samples: u16,
    pub refract_samples: u16,
}

pub fn rgba_to_vector(color: Color) -> Vector<4> {
    vector![color.r, color.g, color.b, color.a] / 255.0
}

pub fn vector_to_rgba(mut vector: Vector<4>) -> Color {
    vector *= 255;
    Color { r: vector[0] as u8, g: vector[1] as u8, b: vector[2] as u8, a: vector[3] as u8 }
}

pub fn rgb_to_vector(color: Color) -> Vector<3> {
    vector![color.r, color.g, color.b] / 255.0
}

pub fn vector_to_rgb(mut vector: Vector<3>) -> Color {
    vector *= 255;
    Color { r: vector[0] as u8, g: vector[1] as u8, b: vector[2] as u8, a: 255u8 }
}

impl Tracer {
    pub fn default() -> Tracer {
        Tracer {
            max_reflect:     8,
            max_refract:     8,
            fov:             90.0,
            reflect_samples: 4,
            refract_samples: 4,
        }
    }
    
    /// Get the closest intersection with a ray, if any.
    pub fn get_intersection(&self, scene: &Scene, ray: Ray) -> Option<Intersect> {
        let mut out: Option<Intersect> = None;
        for i in 0..scene.objects.len() {
            if let Some(intersect) = scene.objects[i].intersect(&ray) {
                if let Some(cur) = out {
                    if cur.distance > intersect.distance {
                        out = Some(intersect);
                    }
                } else {
                    out = Some(intersect);
                }
            }
        }
        out
    }
    
    /// Perform a single sample of ray tracing.
    pub fn trace_single_ray(&self, scene: &Scene, mut ray: Ray, rng: &mut ThreadRng) -> RayTraceResult {
        let mut result = RayTraceResult {
            color:       vector![0, 0, 0],
            did_reflect: false,
            did_refract: false,
        };
        let mut color_mask = vector![1, 1, 1];
        let mut reflect    = self.max_reflect;
        loop {
            if let Some(intersect) = self.get_intersection(scene, ray) {
                // Ray hit an object; decide what to do next.
                result.color += color_mask * intersect.prop.emission * -intersect.normal.dot(ray.normal);
                color_mask *= intersect.prop.color;
                
                // Limit bounce count.
                reflect -= 1;
                result.did_reflect = true;
                if reflect == 0 {
                    return result;
                }
                
                // Determine reflection angle.
                let diff_normal = (Vector::<3>::random_hemisphere_vector(rng, intersect.normal) + intersect.normal).as_unit_vector();
                let spec_normal = (ray.normal - intersect.normal * (2.0 * ray.normal.dot(intersect.normal))).as_unit_vector();
                ray.pos    = intersect.pos;
                ray.normal = spec_normal + (diff_normal - spec_normal) * intersect.prop.roughness;
                
            } else {
                // Ray did not hit anything, get sky color and finish.
                if ray.normal.dot(scene.sun_direction) >= scene.sun_radius {
                    result.color += color_mask * scene.sun_color;
                } else {
                    result.color += color_mask * scene.skybox_color;
                }
                return result;
            };
        }
    }
    
    /// Perform multiple samples of ray tracing.
    pub fn trace_multi_ray(&self, scene: &Scene, ray: Ray, rng: &mut ThreadRng) -> RayTraceResult {
        let mut tmp = self.trace_single_ray(scene, ray, rng);
        let samples = tmp.did_reflect as u16 * self.reflect_samples
                    + tmp.did_refract as u16 * self.refract_samples;
        for _ in 0..samples {
            tmp.color += self.trace_single_ray(scene, ray, rng).color;
        }
        tmp.color /= (samples + 1) as f64;
        tmp
    }
    
    /// Ray-trace an entire image.
    pub fn trace_image(&self, scene: &Scene, fb: &mut dyn Framebuffer, camera: &Transform, rng: &mut ThreadRng) {
        let width    = fb.width();
        let height   = fb.height();
        let fov      = self.fov.to_radians() * 0.5;
        let distance = 0.5 / fov.tan() * width as f64;
        for y in 0..height {
            for x in 0..width {
                let ray = camera.ray_local_to_world(Ray {
                    pos:    vector![0, 0, 0],
                    normal: vector![
                        x as i32 - width as i32 / 2,
                        y as i32 - height as i32 / 2,
                        distance
                    ].as_unit_vector(),
                });
                fb.set_pixel(x, y, self.trace_multi_ray(scene, ray, rng).color);
            }
        }
    }
}

pub struct RayTraceResult {
    pub color:       Vector<3>,
    pub did_reflect: bool,
    pub did_refract: bool,
}

pub struct AsyncTracer {
    threads: Vec<JoinHandle<()>>,
    scene:   Scene,
    tracer:  Tracer,
}

impl AsyncTracer {
    /// Create new async ray tracer.
    pub fn new(num_threads: usize) -> Arc<Mutex<AsyncTracer>> {
        // Created an Arc Mutex so that this can be shared with threads.
        let out = Arc::new(Mutex::new(AsyncTracer {
            threads: Vec::new(),
            scene:   Scene::empty(),
            tracer:  Tracer::default()
        }));
        
        // Instantiate threads.
        let mut ctx = out.lock().unwrap();
        for _ in 0..num_threads {
            // Each thread gets a copy of the Arc.
            let copy = out.clone();
            ctx.borrow_mut().threads.push(spawn(move || {
                AsyncTracer::thread_func(copy);
            }));
        }
        drop(ctx);
        
        out
    }
    
    /// Thread function for async ray tracing.
    fn thread_func(ctx: Arc<Mutex<AsyncTracer>>) {}
    
    /// Perform an async ray tracening.
    pub fn async_trace_image(&mut self, scene: &Scene, camera: &Transform) {}
}



pub trait Framebuffer {
    fn width(&self) -> u16;
    fn height(&self) -> u16;
    fn set_pixel(&mut self, x: u16, y: u16, col: Vector<3>);
}

pub struct SmoothingFramebuffer {
    buffer: Vec<Vector<3>>,
    frame:  u16,
    width:  u16,
    height: u16,
}

impl SmoothingFramebuffer {
    pub fn new(width: u16, height: u16) -> SmoothingFramebuffer {
        SmoothingFramebuffer {
            buffer: vec![vector![0, 0, 0]; width as usize * height as usize],
            frame:  0,
            width:  width,
            height: height,
        }
    }
    
    pub fn update(&mut self, out: &mut dyn Framebuffer) {
        self.frame += 1;
        let scale = 1.0 / self.frame as f64;
        for y in 0..self.height {
            for x in 0..self.width {
                let col = self.buffer[y as usize * self.width as usize + x as usize];
                out.set_pixel(x, y, col * scale);
            }
        }
    }
    
    pub fn get_frame(&self) -> u16 {
        self.frame
    }
}

impl Framebuffer for SmoothingFramebuffer {
    fn width(&self) -> u16 {
        self.width
    }
    fn height(&self) -> u16 {
        self.height
    }
    fn set_pixel(&mut self, x: u16, y: u16, col: Vector<3>) {
        self.buffer[y as usize * self.width as usize + x as usize] += col;
    }
}

impl<T: RenderTarget> Framebuffer for Canvas<T> {
    fn width(&self) -> u16 {
        self.output_size().unwrap().0 as u16
    }
    fn height(&self) -> u16 {
        self.output_size().unwrap().1 as u16
    }
    fn set_pixel(&mut self, x: u16, y: u16, col: Vector<3>) {
        self.set_draw_color(vector_to_rgb(col));
        let _ = self.draw_point(Point::new(x as i32, y as i32));
    }
}
