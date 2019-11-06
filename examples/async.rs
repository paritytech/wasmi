// In this example we execute a module that imports an asynchronous host method
// using tokio::runtime::current_thread which allows async on one the main thread
extern crate wasmi;

use wasmi::*;

use std::rc::Rc;

extern crate tokio;
use std::time::{Duration, Instant};
use tokio::prelude::*;
use tokio::runtime::current_thread::*;
use tokio::timer::{Delay, Interval};

fn main() {
    run(future::lazy(move || {
        //load the prepared wasm
        let buf = include_bytes!("async/untouched.wasm").to_vec();
        let module = Module::from_buffer(buf).unwrap();
        //make an imports builder with an instance of Env
        let environment = Env {};
        let imports = ImportsBuilder::new().with_resolver("env", &environment);
        //start 10 long running tasks
        for x in 0..10 {
            let module_instance = make_module(&module, &imports);
            //spawn future returned by invoke_export_async onto current thread
            //pass in the instance number (x) which will also be how long it will 'take' in secs
            spawn(future::lazy(move || {
                println!("Starting {}", x);
                module_instance
                    .invoke_export_async("run_takes_a_while", vec![RuntimeValue::I32(x)], Rc::new(Env {}))
                    .then(move |r| {
                        println!("Done {} {}", x, r.unwrap().unwrap().try_into::<i32>().unwrap()); //ignore errors for example
                        future::ok(())
                    })
            }));
        }
        //make a task that runs takes less time
        let module_instance = make_module(&module, &imports);
        let mut count:i32=0;
        let repeat = Interval::new_interval(Duration::from_secs(1))
            .take(10)
            .map_err(|e| panic!("Interval failed; err={:?}", e))
            .for_each(move |_| {
                let res=module_instance
                    .invoke_export_async("run_happens_fast", vec![RuntimeValue::I32(2),RuntimeValue::I32(2)], Rc::new(Env {}))
                    .then(move |r| {
                        println!("Happens fast Done {} 2+2={}", count,r.unwrap().unwrap().try_into::<i32>().unwrap()); //ignore errors for example
                        future::ok::<(), ()>(())
                    });
                    count+=1;
                    res
            }
            );
        spawn(repeat);
        future::ok(())
    }));
}

fn make_module<I>(m: &Module, i: &I) -> ModuleRef
where
    I: ImportResolver,
{
    ModuleInstance::new(m, i)
        .expect("Failed to instantiate module")
        .run_start(&mut NopExternals)
        .expect("Failed to run start function in module")
}

struct Env {}

impl ModuleImportResolver for Env {
    fn resolve_func(&self, field_name: &str, _signature: &Signature) -> Result<FuncRef, Error> {
        println!("sig {} {:?}",field_name,_signature);
        match field_name {
            "takes_a_while" => Ok(FuncInstance::alloc_host(
                Signature::new(&[ValueType::I32][..], Some(ValueType::I32)),
                0,
            )),
            "happens_fast" => Ok(FuncInstance::alloc_host(
                Signature::new(&[ValueType::I32, ValueType::I32][..], Some(ValueType::I32)),
                1,
            )),
            _ => {
                return Err(Error::Instantiation(format!(
                    "Export {} not found",
                    field_name
                )))
            }
        }
    }
}

impl AsyncExternals for Env {
    fn invoke_index_async<'b>(
        &self,
        index: usize,
        _args: Vec<RuntimeValue>,
    ) -> Box<dyn Future<Item = Option<RuntimeValue>, Error = Trap> + 'b> {
        match index {
            0 => {
                //takes_a_while
                let args = RuntimeArgs::from(&_args);
                let i = args.nth_checked::<i32>(0).unwrap_or(1);
                let when = Instant::now() + Duration::from_secs(i as u64);
                Box::new(
                    Delay::new(when)
                        .map_err(|e| panic!("timer failed; err={:?}", e))
                        .and_then(move |_| future::ok(Some(RuntimeValue::I32(i)))),
                )
            },
            1 => {
                //happens_fast
                let args = RuntimeArgs::from(&_args);
                let a = args.nth_checked::<i32>(0).unwrap_or(2);
                let b = args.nth_checked::<i32>(1).unwrap_or(2);
                Box::new(future::ok(Some(RuntimeValue::I32(a + b))))
            }
            _ => panic!("Unimplemented function at {}", index),
        }
    }
}
