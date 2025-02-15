use std::{
    borrow::BorrowMut,
    f64::consts::PI,
    ops::{Deref, DerefMut},
    sync::{Arc, Condvar, Mutex},
    thread::{spawn, JoinHandle},
};

use rand::{rngs::ThreadRng, Rng, RngCore};
use sdl2::{
    pixels::Color,
    rect::Point,
    render::{Canvas, RenderTarget},
};

use crate::matrix::*;
use crate::scene::*;
use crate::*;

pub struct Tracer {
    pub max_reflect: u16,
    pub max_refract: u16,
    pub fov: f64,
    pub reflect_samples: u16,
    pub refract_samples: u16,
}

pub fn rgba_to_vector(color: Color) -> Vector<4> {
    vector![color.r, color.g, color.b, color.a] / 255.0
}

pub fn vector_to_rgba(mut vector: Vector<4>) -> Color {
    vector *= 255;
    Color {
        r: vector[0] as u8,
        g: vector[1] as u8,
        b: vector[2] as u8,
        a: vector[3] as u8,
    }
}

pub fn rgb_to_vector(color: Color) -> Vector<3> {
    vector![color.r, color.g, color.b] / 255.0
}

pub fn vector_to_rgb(mut vector: Vector<3>) -> Color {
    vector *= 255;
    Color {
        r: vector[0] as u8,
        g: vector[1] as u8,
        b: vector[2] as u8,
        a: 255u8,
    }
}

impl Tracer {
    pub fn default() -> Tracer {
        Tracer {
            max_reflect: 8,
            max_refract: 8,
            fov: 90.0,
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
    pub fn trace_single_ray(
        &self,
        scene: &Scene,
        mut ray: Ray,
        rng: &mut ThreadRng,
    ) -> RayTraceResult {
        let mut result = RayTraceResult {
            color: vector![0, 0, 0],
            did_reflect: false,
            did_refract: false,
        };
        let mut color_mask = vector![1, 1, 1];
        let mut reflect = self.max_reflect;
        loop {
            if let Some(intersect) = self.get_intersection(scene, ray) {
                // Ray hit an object; decide what to do next.
                result.color += color_mask * intersect.prop.emission;
                color_mask *= intersect.prop.color;

                // Limit bounce count.
                reflect -= 1;
                result.did_reflect = true;
                if reflect == 0 {
                    return result;
                }

                // Choose between reflection and refraction.
                let refract_rng = rng.gen::<f64>();
                if !intersect.is_entry || refract_rng > intersect.prop.opacity {
                    // Determine refraction angle.
                    let (ior0, ior1, normal) = if intersect.is_entry {
                        (1.0, intersect.prop.ior, -intersect.normal)
                    } else {
                        (intersect.prop.ior, 1.0, intersect.normal)
                    };
                    let ratio = ior0 / ior1;
                    let dot = ray.normal.dot(normal);
                    ray.pos = intersect.pos;
                    ray.normal = ray.normal * ratio
                        + normal * ((1.0 - ratio * ratio * (1.0 - dot * dot)).sqrt() - ratio * dot);
                } else {
                    // Determine reflection angle.
                    let diff_normal =
                        (Vector::<3>::random_hemisphere_vector(rng, intersect.normal)
                            + intersect.normal)
                            .as_unit_vector();
                    let spec_normal = (ray.normal
                        - intersect.normal * (2.0 * ray.normal.dot(intersect.normal)))
                    .as_unit_vector();
                    ray.pos = intersect.pos;
                    ray.normal =
                        spec_normal + (diff_normal - spec_normal) * intersect.prop.roughness;
                }
            } else {
                // Ray did not hit anything, get sky color and finish.
                let mut coeff = ray.normal[1] * 3.0;
                coeff = coeff.clamp(-1.0, 1.0);
                let base = if coeff >= 0.0 {
                    scene.horizon_color + (scene.ground_color - scene.horizon_color) * coeff
                } else {
                    scene.horizon_color + (scene.skybox_color - scene.horizon_color) * -coeff
                };
                let sun_dot = ray.normal.dot(scene.sun_direction);
                if sun_dot >= scene.sun_radius {
                    let sun_coeff = (sun_dot - scene.sun_radius) / (1.0 - scene.sun_radius);
                    result.color += color_mask * (base + (scene.sun_color - base) * sun_coeff);
                } else {
                    result.color += color_mask * base;
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

    /// Ray-trace an image with multiple threads.
    pub fn trace_image_async(
        self: &Arc<Self>,
        scene: Arc<Scene>,
        fb: &mut dyn Framebuffer,
        camera: &Transform,
        num_threads: u16,
    ) {
        let bounds = (0, 0, fb.width(), fb.height());

        let mut handles = vec![];
        let mut partial = vec![];
        for i in 0..num_threads {
            let fb = Arc::new(Mutex::new(PartialFramebuffer::new(
                fb.width(),
                fb.height(),
                num_threads,
                i,
            )));
            partial.push(fb.clone());
            let camera = *camera;
            let self2 = self.clone();
            let scene = scene.clone();
            handles.push(spawn(move || {
                let mut rng = thread_rng();
                self2.trace_partial_image(
                    scene.as_ref(),
                    fb.lock().unwrap().deref_mut(),
                    &camera,
                    &mut rng,
                    num_threads,
                    i,
                    bounds,
                );
            }));
        }

        for handle in handles {
            handle.join().unwrap();
        }
        for part in partial {
            part.lock().unwrap().update(fb);
        }
    }

    /// Ray-trace an entire image.
    pub fn trace_image(
        &self,
        scene: &Scene,
        fb: &mut dyn Framebuffer,
        camera: &Transform,
        rng: &mut ThreadRng,
    ) {
        let bounds = (0, 0, fb.width(), fb.height());
        self.trace_partial_image(scene, fb, camera, rng, 1, 0, bounds);
    }

    /// Ray-trace part of an image.
    /// Pixels are interlaced by numeric index.
    /// `bounds` describes an X, Y, width, height rectangle within the framebuffer.
    pub fn trace_partial_image(
        &self,
        scene: &Scene,
        fb: &mut dyn Framebuffer,
        camera: &Transform,
        rng: &mut ThreadRng,
        interlace_amount: u16,
        interlace_offset: u16,
        bounds: (u16, u16, u16, u16),
    ) {
        let width = fb.width();
        let height = fb.height();
        let fov = self.fov.to_radians() * 0.5;
        let distance = 0.5 / fov.tan() * width as f64;
        for y in bounds.1..(bounds.1 + bounds.3) {
            for x in bounds.0..(bounds.0 + bounds.2) {
                if (x as usize + y as usize * width as usize) % interlace_amount as usize
                    != interlace_offset as usize
                {
                    continue;
                }
                let rand_x = rng.next_u32() as f64 / (1u64 << 32) as f64 - 0.5;
                let rand_y = rng.next_u32() as f64 / (1u64 << 32) as f64 - 0.5;
                let ray = camera.ray_local_to_world(Ray {
                    pos: vector![0, 0, 0],
                    normal: vector![
                        rand_x + x as f64 - width as f64 * 0.5,
                        rand_y + y as f64 - height as f64 * 0.5,
                        distance
                    ]
                    .as_unit_vector(),
                });
                fb.set_pixel(x, y, self.trace_multi_ray(scene, ray, rng).color);
            }
        }
    }
}

pub struct RayTraceResult {
    pub color: Vector<3>,
    pub did_reflect: bool,
    pub did_refract: bool,
}

pub trait Framebuffer {
    fn width(&self) -> u16;
    fn height(&self) -> u16;
    fn set_pixel(&mut self, x: u16, y: u16, col: Vector<3>);
}

pub struct PartialFramebuffer {
    data: Vec<[f64; 3]>,
    width: u16,
    height: u16,
    interlace_count: u16,
    interlace_offset: u16,
}

impl PartialFramebuffer {
    pub fn new(
        width: u16,
        height: u16,
        interlace_count: u16,
        interlace_offset: u16,
    ) -> PartialFramebuffer {
        let mut length = width as usize * height as usize;
        if (length % interlace_count as usize) > interlace_offset as usize {
            length = length / (interlace_count as usize) + 1;
        } else {
            length /= interlace_count as usize;
        }
        PartialFramebuffer {
            data: vec![[0f64; 3]; length],
            width,
            height,
            interlace_count,
            interlace_offset,
        }
    }

    pub fn update(&self, other: &mut dyn Framebuffer) {
        assert_eq!(self.width, other.width());
        assert_eq!(self.height, other.height());
        let mut length = self.width as usize * self.height as usize;
        if (length % self.interlace_count as usize) > self.interlace_offset as usize {
            length = length / (self.interlace_count as usize) + 1;
        } else {
            length /= self.interlace_count as usize;
        }
        for i in 0..length {
            let index = i * self.interlace_count as usize + self.interlace_offset as usize;
            let x = (index % self.width as usize) as u16;
            let y = (index / self.width as usize) as u16;
            other.set_pixel(x, y, Vector::from(self.data[i]));
        }
    }
}

impl Framebuffer for PartialFramebuffer {
    fn width(&self) -> u16 {
        self.width
    }

    fn height(&self) -> u16 {
        self.height
    }

    fn set_pixel(&mut self, x: u16, y: u16, col: Vector<3>) {
        let mut index = x as usize + y as usize * self.width as usize;
        if index % self.interlace_count as usize != self.interlace_offset as usize {
            return;
        }
        index /= self.interlace_count as usize;
        self.data[index] = col.data();
    }
}

pub struct SmoothingFramebuffer {
    buffer: Vec<Vector<3>>,
    frame: u16,
    width: u16,
    height: u16,
}

impl SmoothingFramebuffer {
    pub fn new(width: u16, height: u16) -> SmoothingFramebuffer {
        SmoothingFramebuffer {
            buffer: vec![vector![0, 0, 0]; width as usize * height as usize],
            frame: 0,
            width: width,
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
        assert!(x <= self.width);
        assert!(y <= self.height);
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
