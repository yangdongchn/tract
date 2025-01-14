use ndarray::*;

use crate::broadcast::multi_broadcast;
use crate::internal::*;

element_map!(Not, [bool], |a: bool| !a);

element_bin!(And, [bool, u8, i8, i16, i32, i64]
             { |a, b| (a as i64 != 0 && b as i64 != 0) as _});
element_bin!(Or, [bool, u8, i8, i16, i32, i64]
             { |a, b| (a as i64 != 0 || b as i64 != 0) as _});
element_bin!(Xor, [bool] { |a, b| a ^ b });

element_bin!(Equals, [bool, u8, i8, i16, i32, i64, f32, f64, TDim] => bool { |a,b| a==b });
element_bin!(Lesser, [u8, i8, i16, i32, i64, f32, f64] => bool { |a,b| a<b });
element_bin!(LesserEqual, [u8, i8, i16, i32, i64, f32, f64] => bool { |a,b| a<=b });
element_bin!(Greater, [u8, i8, i16, i32, i64, f32, f64] => bool { |a,b| a>b });
element_bin!(GreaterEqual, [u8, i8, i16, i32, i64, f32, f64] => bool { |a,b| a>=b });

#[derive(Debug, Clone, new, Default)]
pub struct Iff;

impl Iff {
    pub fn eval_t<T: Datum>(
        shape: &[usize],
        cond: &ArrayViewD<bool>,
        t: Arc<Tensor>,
        f: Arc<Tensor>,
    ) -> TractResult<Tensor> {
        let mut result = unsafe { Tensor::uninitialized::<T>(shape)? };
        Zip::from(result.to_array_view_mut::<T>()?)
            .and_broadcast(cond)
            .and_broadcast(t.to_array_view::<T>()?)
            .and_broadcast(f.to_array_view::<T>()?)
            .apply(|r, c, t, f|  *r = if *c { t.clone() } else { f.clone() });
        Ok(result)
    }
}

impl Op for Iff {
    fn name(&self) -> Cow<str> {
        "Iff".into()
    }
}

impl StatelessOp for Iff {
    fn eval(&self, mut inputs: TVec<Arc<Tensor>>) -> TractResult<TVec<Arc<Tensor>>> {
        let (cond, t, f) = args_3!(inputs);
        let shape: TVec<usize> = multi_broadcast(&[cond.shape(), t.shape(), f.shape()])
            .ok_or_else(|| {
                format!(
                    "Incompatible shapes {:?}, {:?} and {:?}",
                    cond.shape(),
                    t.shape(),
                    f.shape()
                )
            })?;
        let cond = cond.to_array_view::<bool>()?;
        let c = dispatch_datum!(Self::eval_t(t.datum_type())(&*shape, &cond, t, f))?;
        Ok(tvec!(c.into_arc_tensor()))
    }
}

impl InferenceRulesOp for Iff {
    fn rules<'r, 'p: 'r, 's: 'r>(
        &'s self,
        s: &mut Solver<'r>,
        inputs: &'p [TensorProxy],
        outputs: &'p [TensorProxy],
    ) -> InferenceResult {
        check_input_arity(&inputs, 3)?;
        check_output_arity(&outputs, 1)?;
        s.equals(&inputs[0].datum_type, DatumType::Bool)?;
        s.equals(&inputs[1].datum_type, &outputs[0].datum_type)?;
        s.equals(&inputs[2].datum_type, &outputs[0].datum_type)?;
        s.given_3(&inputs[0].shape, &inputs[1].shape, &inputs[2].shape, move |s, c, t, f| {
            let shape = multi_broadcast(&[&c, &t, &f])
                .ok_or_else(|| format!("Incompatible shapes {:?}, {:?} and {:?}", c, t, f))?;
            s.equals(&outputs[0].shape, shape)
        })?;
        Ok(())
    }

    inference_op_as_op!();
}
