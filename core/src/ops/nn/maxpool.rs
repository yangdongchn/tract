use crate::internal::*;
use ndarray::prelude::*;

use super::{DataFormat, PaddingSpec, Patch};

#[derive(Debug, Clone, new, Default)]
pub struct MaxPool {
    data_fmt: DataFormat,
    kernel_shape: TVec<usize>,
    padding: PaddingSpec,
    strides: Option<TVec<usize>>,
    with_index_outputs: Option<DatumType>,
}

impl MaxPool {
    fn patch(&self, input_full_shape: &[usize]) -> Patch {
        let shape = self.data_fmt.shape(input_full_shape);
        Patch::new(
            tvec![1; shape.hw_rank()],
            self.kernel_shape.clone(),
            &self.padding,
            self.strides.clone().unwrap_or_else(|| tvec![1; shape.hw_rank()]),
            shape.hw_dims().into(),
            shape.hw_stride(),
        )
    }
}

impl Op for MaxPool {
    fn name(&self) -> Cow<str> {
        "MaxPool".into()
    }
}

impl StatelessOp for MaxPool {
    fn eval(&self, mut inputs: TVec<SharedTensor>) -> TractResult<TVec<SharedTensor>> {
        let input = args_1!(inputs);
        let input: ArrayViewD<f32> = input.to_array_view()?;

        let patch = self.patch(input.shape());
        unimplemented!();
        /*
        let shape = self.data_fmt.shape(input.shape());
        let mut output_shape = input.shape().clone();
        output_shape[shape.hw_axes()].copy_from_slice(&*patch.output_spatial_shape);

        let mut values = unsafe { ArrayD::<f32>::uninitialized(&*output_shape) };
        let mut indices = if self.with_index_outputs.is_some() {
            Some(unsafe { ArrayD::<i32>::uninitialized(&*output_shape) })
        } else {
            None
        };
        let c_stride = input.strides()[shape.c_axis()] as isize;
        for i in 0..shape.n() {
            let input = input.slice_axis(Axis(shape.n_axis()), (i..=i).into()).as_ptr();
            for (center, hint) in patch.visit_all() {
            }
            for c in 0..shape.c() as isize {
                let input = input.offset(c * c_stride);
                for 
            }
        }
        ::ndarray::indices(&*output_shape).into_iter().for_each(|coords| {
            let max = patch
                .at(&coords.slice())
                .enumerate()
                .filter_map(|(ix, v)| v.map(|v| (ix, v)))
                .fold((0, ::std::f32::MIN), |acc, v| if acc.1 < v.1 { v } else { acc });
            values[&coords] = max.1;
            if self.with_index_outputs.is_some() {
                indices.as_mut().unwrap()[coords] =
                    visitor.global_offset_for(&coords.slice(), max.0) as i32;
            }
        });
        if let Some(dt) = self.with_index_outputs {
            Ok(tvec!(
                values.into(),
                Tensor::from(indices.unwrap()).cast_to_dt(dt)?.into_owned().into_tensor()
            ))
        } else {
            Ok(tvec!(values.into()))
        }
        */
    }
}

impl InferenceRulesOp for MaxPool {
    fn rules<'r, 'p: 'r, 's: 'r>(
        &'s self,
        s: &mut Solver<'r>,
        inputs: &'p [TensorProxy],
        outputs: &'p [TensorProxy],
    ) -> InferenceResult {
        check_output_arity(&outputs, 1 + self.with_index_outputs.is_some() as usize)?;
        s.equals(&outputs[0].datum_type, &inputs[0].datum_type)?;
        s.equals(&outputs[0].rank, &inputs[0].rank)?;
        if let Some(idt) = self.with_index_outputs {
            s.equals(&outputs[1].datum_type, idt)?;
            s.equals(&outputs[1].rank, &inputs[0].rank)?;
        }
        s.given(&inputs[0].shape, move |s, ishape| {
            let ishape = self.data_fmt.shape(ishape);
            let ones = tvec![1; ishape.hw_rank()];
            let computed = self.padding.compute(
                ishape.hw_dims(),
                &*self.kernel_shape,
                &ones,
                self.strides.as_ref().unwrap_or(&ones),
            );
            for o in 0..outputs.len() {
                for (ix, d) in computed.dims.iter().enumerate() {
                    s.equals(&outputs[o].shape[ix + ishape.h_axis()], d.output)?;
                }
                s.equals(&outputs[o].shape[ishape.n_axis()], ishape.n_dim())?;
                s.equals(&outputs[o].shape[ishape.c_axis()], ishape.c_dim())?;
            }
            Ok(())
        })
    }
}
