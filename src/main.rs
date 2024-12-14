
mod matrix;
mod scene;
mod tracer;
use std::process::exit;

use crate::tracer::*;
use crate::scene::*;
use crate::matrix::*;

use rand::thread_rng;
use sdl2::event::Event;
use sdl2::pixels::Color;

fn main() {
    let sdl_ctx = sdl2::init().unwrap();
    let vid_ctx = sdl_ctx.video().unwrap();
    let window = vid_ctx.window("Ray Tracer", 300, 300).position_centered().build().unwrap();
    let mut canvas = window.into_canvas().build().unwrap();
    let mut event_pump = sdl_ctx.event_pump().unwrap();
    
    let tracer = Tracer::default();
    let scene  = Scene {
        objects: vec![
            Box::new(Sphere {
                transform: Transform::from(vector![0, 0, 2], vector![1, 1, 1], vector![0, 0, 0]),
                radius:    0.5,
                prop:      PhysProp::from_color(vector![1, 0, 0]),
            }),
            Box::new(Sphere {
                transform: Transform::from(vector![-1, 0, 2], vector![1, 1, 1], vector![0, 0, 0]),
                radius:    0.4,
                prop:      PhysProp {
                    color:     vector![0, 1, 0],
                    ior:       1.0,
                    roughness: 0.0,
                    emission:  vector![0, 0, 0],
                },
            }),
            Box::new(Plane {
                transform: Transform::from(vector![0, 0.5, 2], vector![1, 1, 1], vector![90, 0, 0]),
                prop:      PhysProp::from_color(vector![0.5, 0.5, 0.5]),
            }),
            Box::new(Sphere {
                transform: Transform::from(vector![-0.5, 0.3, 1.5], vector![1, 1, 1], vector![0, 0, 0]),
                radius:    0.2,
                prop:      PhysProp::from_emission(vector![1, 1, 0], vector![1, 1, 0]),
            }),
        ],
        skybox_color:  vector![0, 0.2, 0.3],
        sun_color:     vector![2, 2, 1.4],
        sun_direction: vector![1, -1, -1].as_unit_vector(),
        sun_radius:    0.8,
    };
    let camera = Transform::from(vector![0, 0, 0], vector![1, 1, 1], vector![0, 0, 0]);
    
    if let Ok((_,_)) = canvas.output_size() {
    }
    
    let mut buffer = SmoothingFramebuffer::new(canvas.output_size().unwrap().0 as u16, canvas.output_size().unwrap().1 as u16);
    let mut rng    = thread_rng();
    
    'rtx_loop: loop {
        if buffer.get_frame() < u16::MAX {
            canvas.set_draw_color(Color::BLACK);
            canvas.clear();
            tracer.trace_image(&scene, &mut buffer, &camera, &mut rng);
            buffer.update(&mut canvas);
            canvas.present();
        }
        while let Some(event) = event_pump.poll_event() {
            match event {
                Event::Quit{..} => break 'rtx_loop,
                _ => {}
            }
        }
    }
    exit(0);
}
