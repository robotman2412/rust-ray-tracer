use std::f64::consts::TAU;

use rand::{rngs::ThreadRng, Rng};

// Floating-point matrix of fixed size.
#[derive(Clone, Copy, PartialEq)]
pub struct Matrix<const W: usize, const H: usize> {
    data: [[f64; W]; H],
}

// Matrices implement the Eq trait.
impl<const W: usize, const H: usize> Eq for Matrix<W, H> {}

impl<const W: usize, const H: usize> Matrix<W, H> {
    pub fn from(data: [[f64; W]; H]) -> Matrix<W, H> {
        Matrix { data: data }
    }
    pub fn zero() -> Matrix<W, H> {
        Matrix {
            data: [[0.0; W]; H],
        }
    }
    pub fn get(&self, x: usize, y: usize) -> f64 {
        self.data[y][x]
    }
    pub fn set(&mut self, x: usize, y: usize, value: f64) {
        self.data[y][x] = value
    }
}

// Identity matrix constructor.
impl<const D: usize> Matrix<D, D> {
    pub fn identity() -> Matrix<D, D> {
        let mut tmp = Matrix::<D, D>::zero();
        for i in 0..D {
            tmp.data[i][i] = 1.0;
        }
        tmp
    }
}

// Matrix-matrix multiplication function.
impl<const M: usize, const N: usize, const P: usize> std::ops::Mul<Matrix<N, P>> for Matrix<M, N> {
    type Output = Matrix<M, P>;
    fn mul(self, rhs: Matrix<N, P>) -> Matrix<M, P> {
        let mut out = Matrix::<M, P>::zero();
        for j in 0..P {
            for i in 0..M {
                for k in 0..N {
                    out.data[j][i] += self.data[k][i] * rhs.data[j][k];
                }
            }
        }
        out
    }
}

// Matrix-matrix multiplication-assign function.
impl<const D: usize> std::ops::MulAssign<Matrix<D, D>> for Matrix<D, D> {
    fn mul_assign(&mut self, rhs: Matrix<D, D>) {
        *self = *self * rhs;
    }
}

impl Matrix<3, 3> {
    pub fn rotate_x(angle: f64) -> Matrix<3, 3> {
        let (sin, cos) = angle.sin_cos();
        Matrix::from([[1.0, 0.0, 0.0], [0.0, cos, -sin], [0.0, sin, cos]])
    }
    pub fn rotate_y(angle: f64) -> Matrix<3, 3> {
        let (sin, cos) = angle.sin_cos();
        Matrix::from([[cos, 0.0, sin], [0.0, 1.0, 0.0], [-sin, 0.0, cos]])
    }
    pub fn rotate_z(angle: f64) -> Matrix<3, 3> {
        let (sin, cos) = angle.sin_cos();
        Matrix::from([[cos, -sin, 0.0], [sin, cos, 0.0], [0.0, 0.0, 1.0]])
    }
    pub fn rotate_xyz(x: f64, y: f64, z: f64) -> Matrix<3, 3> {
        Matrix::rotate_x(x) * Matrix::rotate_x(y) * Matrix::rotate_x(z)
    }
    pub fn rotate(angles: Vector<3>) -> Matrix<3, 3> {
        Matrix::rotate_xyz(angles[0], angles[1], angles[2])
    }
    pub fn scale_xyz(x: f64, y: f64, z: f64) -> Matrix<3, 3> {
        Matrix::from([[x, 0.0, 0.0], [0.0, y, 0.0], [0.0, 0.0, y]])
    }
    pub fn scale(coeffs: Vector<3>) -> Matrix<3, 3> {
        Matrix::scale_xyz(coeffs[0], coeffs[1], coeffs[2])
    }
}

// Floating-point vector of fixed size.
#[derive(Clone, Copy, PartialEq)]
pub struct Vector<const L: usize> {
    data: [f64; L],
}

// Vectors implement the Eq trait.
impl<const L: usize> Eq for Vector<L> {}

// Contructors and vector functions.
impl<const L: usize> Vector<L> {
    pub fn new(data: [f64; L]) -> Vector<L> {
        Vector::<L> { data: data }
    }
    pub fn zero() -> Vector<L> {
        Vector::<L> { data: [0.0; L] }
    }
    pub fn from<T: AsF64 + Copy>(data: [T; L]) -> Vector<L> {
        Vector::<L> {
            data: data.map(|i| i.to_f64()),
        }
    }
    pub fn data(&self) -> [f64; L] {
        self.data
    }
    pub fn sqr_magnitude(&self) -> f64 {
        self.data.iter().map(|f| f * f).sum()
    }
    pub fn magnitude(&self) -> f64 {
        self.sqr_magnitude().sqrt()
    }
    pub fn as_unit_vector(&self) -> Vector<L> {
        *self / self.magnitude()
    }
    pub fn to_unit_vector(&mut self) {
        *self /= self.magnitude();
    }
    pub fn dot(&self, other: Vector<L>) -> f64 {
        let mut sum = 0.0;
        for i in 0..L {
            sum += self[i] * other[i];
        }
        sum
    }
    /// Random unit vector.
    pub fn random_unit_vector(rng: &mut ThreadRng) -> Vector<L> {
        let mut tmp = [0.0; L];
        for i in 0..L {
            tmp[i] = random_normal(rng);
        }
        Vector::from(tmp).as_unit_vector()
    }
    /// Random unit vector in a hemisphere.
    pub fn random_hemisphere_vector(rng: &mut ThreadRng, relative_to: Vector<L>) -> Vector<L> {
        let tmp = Vector::random_unit_vector(rng);
        if tmp.dot(relative_to) < 0.0 {
            -tmp
        } else {
            tmp
        }
    }
}

// Indexing vectors.
impl<const L: usize> std::ops::Index<usize> for Vector<L> {
    type Output = f64;
    fn index<'a>(&'a self, i: usize) -> &'a f64 {
        &self.data[i]
    }
}

impl<const L: usize> std::ops::IndexMut<usize> for Vector<L> {
    fn index_mut<'a>(&'a mut self, i: usize) -> &'a mut f64 {
        &mut self.data[i]
    }
}

// Vector-vector operators.
macro_rules! vv_op {
    ($trait:ident, $func:ident, $trait_assign:ident, $func_assign:ident, $op:tt) => {
        // Vector-vector infix.
        impl<const L: usize> std::ops::$trait<Vector<L>> for Vector<L> {
            type Output = Vector<L>;
            fn $func(self, rhs: Vector<L>) -> Vector<L> {
                let mut tmp = Vector::<L>::zero();
                for i in 0..L { tmp[i] = self[i] $op rhs[i]; }
                tmp
            }
        }
        // Vector-vector assign.
        impl<const L: usize> std::ops::$trait_assign<Vector<L>> for Vector<L> {
            fn $func_assign(&mut self, rhs: Vector<L>) {
                for i in 0..L { self[i] = self[i] $op rhs[i]; }
            }
        }
    }
}
vv_op!(Add, add, AddAssign, add_assign, +);
vv_op!(Sub, sub, SubAssign, sub_assign, -);
vv_op!(Mul, mul, MulAssign, mul_assign, *);
vv_op!(Div, div, DivAssign, div_assign, /);

// Vector-float operators.
macro_rules! vf_op {
    ($trait:ident, $func:ident, $trait_assign:ident, $func_assign:ident, $op:tt) => {
        // Vector-float infix.
        impl<const L: usize, T: AsF64+Copy> std::ops::$trait<T> for Vector<L> {
            type Output = Vector<L>;
            fn $func(self, rhs: T) -> Vector<L> {
                let rhs_tmp = rhs.to_f64();
                let mut tmp = Vector::<L>::zero();
                for i in 0..L { tmp[i] = self[i] $op rhs_tmp; }
                tmp
            }
        }
        // Vector-float assign.
        impl<const L: usize, T: AsF64+Copy> std::ops::$trait_assign<T> for Vector<L> {
            fn $func_assign(&mut self, rhs: T) {
                let rhs_tmp = rhs.to_f64();
                for i in 0..L { self[i] = self[i] $op rhs_tmp; }
            }
        }
    };
}
vf_op!(Mul, mul, MulAssign, mul_assign, *);
vf_op!(Div, div, DivAssign, div_assign, /);

// Vector-matrix multiplication.
impl<const W: usize, const H: usize> std::ops::Mul<Matrix<W, H>> for Vector<W> {
    type Output = Vector<H>;
    fn mul(self, rhs: Matrix<W, H>) -> Vector<H> {
        let mut tmp = Vector::<H>::zero();
        for i in 0..H {
            for j in 0..W {
                tmp[i] += rhs.data[j][i] * self[j];
            }
        }
        tmp
    }
}
impl<const D: usize> std::ops::MulAssign<Matrix<D, D>> for Vector<D> {
    fn mul_assign(&mut self, rhs: Matrix<D, D>) {
        *self = *self * rhs;
    }
}

impl<const L: usize> std::ops::Neg for Vector<L> {
    type Output = Vector<L>;
    fn neg(self) -> Vector<L> {
        Vector {
            data: self.data.map(f64::neg),
        }
    }
}

// Helpers for converting into f64.
pub trait AsF64 {
    fn to_f64(self) -> f64;
}
impl AsF64 for f32 {
    fn to_f64(self) -> f64 {
        self as f64
    }
}
impl AsF64 for f64 {
    fn to_f64(self) -> f64 {
        self
    }
}
impl AsF64 for isize {
    fn to_f64(self) -> f64 {
        self as f64
    }
}
impl AsF64 for usize {
    fn to_f64(self) -> f64 {
        self as f64
    }
}
impl AsF64 for i128 {
    fn to_f64(self) -> f64 {
        self as f64
    }
}
impl AsF64 for u128 {
    fn to_f64(self) -> f64 {
        self as f64
    }
}
impl AsF64 for i64 {
    fn to_f64(self) -> f64 {
        self as f64
    }
}
impl AsF64 for u64 {
    fn to_f64(self) -> f64 {
        self as f64
    }
}
impl AsF64 for i32 {
    fn to_f64(self) -> f64 {
        self as f64
    }
}
impl AsF64 for u32 {
    fn to_f64(self) -> f64 {
        self as f64
    }
}
impl AsF64 for i16 {
    fn to_f64(self) -> f64 {
        self as f64
    }
}
impl AsF64 for u16 {
    fn to_f64(self) -> f64 {
        self as f64
    }
}
impl AsF64 for i8 {
    fn to_f64(self) -> f64 {
        self as f64
    }
}
impl AsF64 for u8 {
    fn to_f64(self) -> f64 {
        self as f64
    }
}

#[macro_export]
macro_rules! vector {
    ($($x:expr),+) => {
        Vector::from([$(AsF64::to_f64($x)),+])
    };
}

// Random value in normal distribution where mean=1 and sd=1.
pub fn random_normal(rng: &mut ThreadRng) -> f64 {
    let t = TAU * rng.gen::<f64>();
    let r = (rng.gen::<f64>().ln() * -2.0).sqrt();
    r * t.cos()
}
