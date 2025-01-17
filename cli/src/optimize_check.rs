use tract_core::internal::*;
use tract_core::ops::source::Source;

use crate::display_graph;
use crate::{CliResult, Parameters, SomeModel};

pub fn handle(params: Parameters, _options: display_graph::DisplayOptions) -> CliResult<()> {
    let plain = params.typed_model.unwrap();
    let optimized = if let SomeModel::Typed(m) = params.tract_model {
        m
    } else {
        bail!("Can only optimize-check typed models")
    };

    let generated = crate::tensor::make_inputs(&[plain.input_fact(0)?.to_tensor_fact()])?;

    let original_plan = SimplePlan::new(plain)?;
    let mut original_state = SimpleState::new(original_plan)?;
    original_state.set_inputs(generated.clone())?;
    let optimized_plan = SimplePlan::new(optimized)?;
    let mut optimized_state = SimpleState::new(optimized_plan)?;
    optimized_state.set_inputs(generated)?;

    for orig in original_state.plan().order.clone() {
        let optim = {
            let name = &original_state.model().node(orig).name;
            optimized_state.model().node_by_name(name).ok().map(|node| node.id)
        };
        if let Some(optim) = optim {
            if original_state.model().nodes()[orig].op_is::<Source>() {
                continue;
            }
            let orig_result: TVec<_> =
                original_state.compute_recursively(orig)?.into_iter().cloned().collect();
            let optim_result: TVec<_> =
                optimized_state.compute_recursively(optim)?.into_iter().cloned().collect();
            if orig_result.len() != optim_result.len() {
                bail!(
                    "Number of output differ: optimized:{}, original:{}",
                    optim_result.len(),
                    orig_result.len()
                )
            }

            for (got, exp) in optim_result.iter().zip(orig_result.iter()) {
                if !exp.close_enough(got, true) {
                    error!(
                        "Values for {} are not close enough",
                        original_state.model().nodes()[orig]
                    );
                    println!("{:?}\n", original_state.model().nodes()[orig]);
                    println!("{:?}\n", exp);
                    println!("{:?}\n", optimized_state.model().nodes()[optim]);
                    println!("{:?}\n", got);
                    Err("Mismatch")?
                }
            }
            println!("Checked {} - {}", orig, optim);
        } else {
            println!(
                "Could not link node {} to optimized model",
                original_state.model().node(orig)
            );
        }
    }

    info!("Looks good!");
    Ok(())
}
