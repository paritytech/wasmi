(module
 (type $FUNCSIG$ii (func (param i32) (result i32)))
 (type $FUNCSIG$iii (func (param i32 i32) (result i32)))
 (type $FUNCSIG$v (func))
 (import "env" "takes_a_while" (func $assembly/index/takes_a_while (param i32) (result i32)))
 (import "env" "happens_fast" (func $assembly/index/happens_fast (param i32 i32) (result i32)))
 (memory $0 0)
 (table $0 1 funcref)
 (elem (i32.const 0) $null)
 (export "memory" (memory $0))
 (export "run_takes_a_while" (func $assembly/index/run_takes_a_while))
 (export "run_happens_fast" (func $assembly/index/run_happens_fast))
 (func $assembly/index/run_takes_a_while (; 2 ;) (type $FUNCSIG$ii) (param $0 i32) (result i32)
  local.get $0
  call $assembly/index/takes_a_while
 )
 (func $assembly/index/run_happens_fast (; 3 ;) (type $FUNCSIG$iii) (param $0 i32) (param $1 i32) (result i32)
  local.get $0
  local.get $1
  call $assembly/index/happens_fast
 )
 (func $null (; 4 ;) (type $FUNCSIG$v)
 )
)
