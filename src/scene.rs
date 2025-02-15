use sdl2::pixels::Color;

use crate::matrix::*;
use crate::*;

#[derive(Clone, Copy, PartialEq, Eq)]
pub struct Transform {
    pos: Vector<3>,
    scale: Vector<3>,
    angle: Vector<3>,
    mtx: Matrix<3, 3>,
    inv_mtx: Matrix<3, 3>,
}

impl Transform {
    pub fn identity() -> Transform {
        Transform {
            pos: vector![0, 0, 0],
            scale: vector![1, 1, 1],
            angle: vector![0, 0, 0],
            mtx: Matrix::identity(),
            inv_mtx: Matrix::identity(),
        }
    }
    pub fn from(pos: Vector<3>, scale: Vector<3>, angle: Vector<3>) -> Transform {
        let mut tmp = Transform {
            pos: pos,
            scale: scale,
            angle: angle,
            mtx: Matrix::zero(),
            inv_mtx: Matrix::zero(),
        };
        tmp.gen_mtx();
        tmp
    }

    pub fn pos<'a>(&'a self) -> &'a Vector<3> {
        &self.pos
    }
    pub fn set_pos(&mut self, pos: Vector<3>) {
        self.pos = pos;
    }

    pub fn scale<'a>(&'a self) -> &'a Vector<3> {
        &self.scale
    }
    pub fn set_scale(&mut self, scale: Vector<3>) {
        self.scale = scale;
        self.gen_mtx();
    }

    pub fn angle<'a>(&'a self) -> &'a Vector<3> {
        &self.angle
    }
    pub fn set_angle(&mut self, angle: Vector<3>) {
        self.angle = angle;
        self.gen_mtx();
    }

    fn gen_mtx(&mut self) {
        self.mtx = Matrix::rotate_x(self.angle[0].to_radians())
            * Matrix::rotate_y(self.angle[1].to_radians())
            * Matrix::rotate_z(self.angle[2].to_radians());
        self.inv_mtx = Matrix::rotate_z(-self.angle[2].to_radians())
            * Matrix::rotate_y(-self.angle[1].to_radians())
            * Matrix::rotate_x(-self.angle[0].to_radians());
    }

    pub fn world_to_local(&self, mut pos: Vector<3>) -> Vector<3> {
        pos -= self.pos;
        pos *= self.inv_mtx;
        pos /= self.scale;
        pos
    }
    pub fn local_to_world(&self, mut pos: Vector<3>) -> Vector<3> {
        pos *= self.scale;
        pos *= self.mtx;
        pos += self.pos;
        pos
    }

    pub fn normal_world_to_local(&self, normal: Vector<3>) -> Vector<3> {
        normal * self.inv_mtx
    }
    pub fn normal_local_to_world(&self, normal: Vector<3>) -> Vector<3> {
        normal * self.mtx
    }

    pub fn ray_world_to_local(&self, ray: Ray) -> Ray {
        Ray {
            pos: self.world_to_local(ray.pos),
            normal: self.normal_world_to_local(ray.normal),
        }
    }
    pub fn ray_local_to_world(&self, ray: Ray) -> Ray {
        Ray {
            pos: self.local_to_world(ray.pos),
            normal: self.normal_local_to_world(ray.normal),
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub struct Ray {
    /// Position of the ray.
    pub pos: Vector<3>,
    /// Direction the ray is facing.
    pub normal: Vector<3>,
}

#[derive(Clone, Copy, PartialEq)]
pub struct PhysProp {
    pub ior: f64,
    pub opacity: f64,
    pub roughness: f64,
    pub color: Vector<3>,
    pub emission: Vector<3>,
}

impl Eq for PhysProp {}

impl PhysProp {
    pub fn from_color(color: Vector<3>) -> PhysProp {
        PhysProp {
            ior: 1.0,
            opacity: 1.0,
            roughness: 1.0,
            color,
            emission: vector![0, 0, 0],
        }
    }
    pub fn from_opacity(color: Vector<3>, opacity: f64) -> PhysProp {
        PhysProp {
            ior: 1.0,
            opacity,
            roughness: 1.0,
            color,
            emission: vector![0, 0, 0],
        }
    }
    pub fn from_emission(color: Vector<3>, emission: Vector<3>) -> PhysProp {
        PhysProp {
            ior: 1.0,
            opacity: 1.0,
            roughness: 1.0,
            color,
            emission,
        }
    }
}

#[derive(Clone, Copy, PartialEq)]
pub struct Intersect {
    /// Intersection position in world space.
    pub pos: Vector<3>,
    /// Surface normal.
    pub normal: Vector<3>,
    /// Physical properties at the intersection.
    pub prop: PhysProp,
    /// Distance from the ray origin.
    pub distance: f64,
    /// Whether the ray started outside the objct.
    pub is_entry: bool,
}
impl Eq for Intersect {}

pub trait Object {
    fn transform<'a>(&'a self) -> &'a Transform;
    fn transform_mut<'a>(&'a mut self) -> &'a mut Transform;
    fn set_transform(&mut self, pos: Transform) {
        *self.transform_mut() = pos;
    }
    /// Perform an intersection test with a ray in world space.
    fn intersect(&self, ray: &Ray) -> Option<Intersect>;
}

pub struct Sphere {
    pub transform: Transform,
    pub radius: f64,
    pub prop: PhysProp,
}

impl Object for Sphere {
    fn transform<'a>(&'a self) -> &'a Transform {
        &self.transform
    }
    fn transform_mut<'a>(&'a mut self) -> &'a mut Transform {
        &mut self.transform
    }

    fn intersect(&self, ray: &Ray) -> Option<Intersect> {
        let ray = self.transform.ray_world_to_local(*ray);
        let a = -ray.normal.dot(ray.pos);
        let b = a * a - ray.pos.sqr_magnitude() + self.radius * self.radius;

        if b < 0.0 {
            return None;
        }
        let distance: f64;
        if b < 0.00000001 {
            if a > 0.00000001 {
                distance = a;
            } else {
                return None;
            }
        } else {
            let dist0 = a + b.sqrt();
            let dist1 = a - b.sqrt();
            if dist0 < dist1 && dist0 > 0.00000001 {
                distance = dist0;
            } else if dist1 > 0.00000001 {
                distance = dist1;
            } else {
                return None;
            }
        };
        let pos = ray.pos + ray.normal * distance;

        return Some(Intersect {
            pos: self.transform.local_to_world(pos),
            normal: self.transform.normal_local_to_world(pos / self.radius),
            prop: self.prop,
            distance,
            is_entry: ray.pos.sqr_magnitude() > self.radius * self.radius,
        });
    }
}

pub struct Plane {
    pub transform: Transform,
    pub prop: PhysProp,
}

impl Object for Plane {
    fn transform<'a>(&'a self) -> &'a Transform {
        &self.transform
    }
    fn transform_mut<'a>(&'a mut self) -> &'a mut Transform {
        &mut self.transform
    }

    fn intersect(&self, ray: &Ray) -> Option<Intersect> {
        let ray = self.transform.ray_world_to_local(*ray);
        if ray.normal[2].abs() < 0.00000001 {
            return None;
        }
        let distance = -ray.pos[2] / ray.normal[2];
        if distance <= 0.00000001 {
            return None;
        }
        let pos = ray.pos + ray.normal * distance;
        if pos[0].abs() > 1.0 || pos[1].abs() > 1.0 {
            return None;
        }
        Some(Intersect {
            pos: self.transform.local_to_world(pos),
            normal: self
                .transform
                .normal_local_to_world(vector![0, 0, ray.pos[2].signum()]),
            prop: self.prop,
            distance,
            is_entry: true,
        })
    }
}

pub struct Scene {
    /// List of objects in the scene.
    pub objects: Vec<Box<dyn Object + Send + Sync>>,
    /// Ground color.
    pub ground_color: Vector<3>,
    /// Horizon color.
    pub horizon_color: Vector<3>,
    /// Skybox color.
    pub skybox_color: Vector<3>,
    /// Sun color.
    pub sun_color: Vector<3>,
    /// Unit vector pointing at the sun.
    pub sun_direction: Vector<3>,
    /// Dot product threshold for a ray to be pointing at the sun.
    pub sun_radius: f64,
}

impl Scene {
    pub fn empty() -> Scene {
        Scene {
            objects: Vec::new(),
            ground_color: vector![0, 0, 0],
            horizon_color: vector![0, 0, 0],
            skybox_color: vector![0, 0, 0],
            sun_color: vector![0, 0, 0],
            sun_direction: vector![0, 0, 0],
            sun_radius: 1.0,
        }
    }
}
