use alloc::{
    borrow::ToOwned,
    rc::Rc,
    string::{String, ToString},
    vec::Vec,
};
use core::cell::RefCell;
use core::fmt;
use Trap;

use alloc::collections::BTreeMap;

use core::cell::Ref;
use func::{FuncBody, FuncInstance, FuncRef};
use global::{GlobalInstance, GlobalRef};
use host::{Externals,AsyncExternals};
use imports::ImportResolver;
use memory::MemoryRef;
use memory_units::Pages;
use parity_wasm::elements::{External, InitExpr, Instruction, Internal, ResizableLimits, Type};
use runner::StackRecycler;
use table::TableRef;
use types::{GlobalDescriptor, MemoryDescriptor, TableDescriptor};
use validation::{DEFAULT_MEMORY_INDEX, DEFAULT_TABLE_INDEX};
use {Error, MemoryInstance, Module, RuntimeValue, Signature, TableInstance};
use futures::{Future, future};

/// Reference to a [`ModuleInstance`].
///
/// This reference has a reference-counting semantics.
///
/// All [`ModuleInstance`] have strong references to it's components (i.e.
/// globals, memories, funcs, tables), however, this components have
/// weak references to it's containing module. This might be a problem
/// at execution time.
///
/// So if have to make sure that all modules which might be needed at execution time
/// should be retained.
///
/// [`ModuleInstance`]: struct.ModuleInstance.html
#[derive(Clone, Debug)]
pub struct ModuleRef(pub(crate) Rc<ModuleInstance>);

impl ::core::ops::Deref for ModuleRef {
    type Target = ModuleInstance;
    fn deref(&self) -> &ModuleInstance {
        &self.0
    }
}

/// An external value is the runtime representation of an entity
/// that can be imported or exported.
pub enum ExternVal {
    /// [Function][`FuncInstance`].
    ///
    /// [`FuncInstance`]: struct.FuncInstance.html
    Func(FuncRef),
    /// [Table][`TableInstance`].
    ///
    /// [`TableInstance`]: struct.TableInstance.html
    Table(TableRef),
    /// [Memory][`MemoryInstance`].
    ///
    /// [`MemoryInstance`]: struct.MemoryInstance.html
    Memory(MemoryRef),
    /// [Global][`GlobalInstance`].
    ///
    /// Should be immutable.
    ///
    /// [`GlobalInstance`]: struct.GlobalInstance.html
    Global(GlobalRef),
}

impl Clone for ExternVal {
    fn clone(&self) -> Self {
        match *self {
            ExternVal::Func(ref func) => ExternVal::Func(func.clone()),
            ExternVal::Table(ref table) => ExternVal::Table(table.clone()),
            ExternVal::Memory(ref memory) => ExternVal::Memory(memory.clone()),
            ExternVal::Global(ref global) => ExternVal::Global(global.clone()),
        }
    }
}

impl fmt::Debug for ExternVal {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "ExternVal {{ {} }}",
            match *self {
                ExternVal::Func(_) => "Func",
                ExternVal::Table(_) => "Table",
                ExternVal::Memory(_) => "Memory",
                ExternVal::Global(_) => "Global",
            }
        )
    }
}

impl ExternVal {
    /// Get underlying function reference if this `ExternVal` contains
    /// a function, or `None` if it is some other kind.
    pub fn as_func(&self) -> Option<&FuncRef> {
        match *self {
            ExternVal::Func(ref func) => Some(func),
            _ => None,
        }
    }

    /// Get underlying table reference if this `ExternVal` contains
    /// a table, or `None` if it is some other kind.
    pub fn as_table(&self) -> Option<&TableRef> {
        match *self {
            ExternVal::Table(ref table) => Some(table),
            _ => None,
        }
    }

    /// Get underlying memory reference if this `ExternVal` contains
    /// a memory, or `None` if it is some other kind.
    pub fn as_memory(&self) -> Option<&MemoryRef> {
        match *self {
            ExternVal::Memory(ref memory) => Some(memory),
            _ => None,
        }
    }

    /// Get underlying global variable reference if this `ExternVal` contains
    /// a global, or `None` if it is some other kind.
    pub fn as_global(&self) -> Option<&GlobalRef> {
        match *self {
            ExternVal::Global(ref global) => Some(global),
            _ => None,
        }
    }
}

/// A module instance is the runtime representation of a [module][`Module`].
///
/// It is created by instantiating a [module][`Module`], and collects runtime representations
/// of all entities that are imported or defined by the module, namely:
///
/// - [functions][`FuncInstance`],
/// - [memories][`MemoryInstance`],
/// - [tables][`TableInstance`],
/// - [globals][`GlobalInstance`],
///
/// In order to instantiate a module you need to provide entities to satisfy
/// every module's imports (i.e. wasm modules don't have optional imports).
///
/// After module is instantiated you can start invoking it's exported functions with [`invoke_export`].
///
/// [`Module`]: struct.Module.html
/// [`FuncInstance`]: struct.FuncInstance.html
/// [`MemoryInstance`]: struct.MemoryInstance.html
/// [`TableInstance`]: struct.TableInstance.html
/// [`GlobalInstance`]: struct.GlobalInstance.html
/// [`invoke_export`]: #method.invoke_export
#[derive(Debug)]
pub struct ModuleInstance {
    signatures: RefCell<Vec<Rc<Signature>>>,
    tables: RefCell<Vec<TableRef>>,
    funcs: RefCell<Vec<FuncRef>>,
    memories: RefCell<Vec<MemoryRef>>,
    globals: RefCell<Vec<GlobalRef>>,
    exports: RefCell<BTreeMap<String, ExternVal>>,
}

impl ModuleInstance {
    fn default() -> Self {
        ModuleInstance {
            funcs: RefCell::new(Vec::new()),
            signatures: RefCell::new(Vec::new()),
            tables: RefCell::new(Vec::new()),
            memories: RefCell::new(Vec::new()),
            globals: RefCell::new(Vec::new()),
            exports: RefCell::new(BTreeMap::new()),
        }
    }

    pub(crate) fn memory_by_index(&self, idx: u32) -> Option<MemoryRef> {
        self.memories.borrow_mut().get(idx as usize).cloned()
    }

    pub(crate) fn table_by_index(&self, idx: u32) -> Option<TableRef> {
        self.tables.borrow_mut().get(idx as usize).cloned()
    }

    pub(crate) fn global_by_index(&self, idx: u32) -> Option<GlobalRef> {
        self.globals.borrow_mut().get(idx as usize).cloned()
    }

    pub(crate) fn func_by_index(&self, idx: u32) -> Option<FuncRef> {
        self.funcs.borrow().get(idx as usize).cloned()
    }

    pub(crate) fn signature_by_index(&self, idx: u32) -> Option<Rc<Signature>> {
        self.signatures.borrow().get(idx as usize).cloned()
    }

    fn push_func(&self, func: FuncRef) {
        self.funcs.borrow_mut().push(func);
    }

    fn push_signature(&self, signature: Rc<Signature>) {
        self.signatures.borrow_mut().push(signature)
    }

    fn push_memory(&self, memory: MemoryRef) {
        self.memories.borrow_mut().push(memory)
    }

    fn push_table(&self, table: TableRef) {
        self.tables.borrow_mut().push(table)
    }

    fn push_global(&self, global: GlobalRef) {
        self.globals.borrow_mut().push(global)
    }

    /// Access all globals. This is a non-standard API so it's unlikely to be
    /// portable to other engines.
    pub fn globals<'a>(&self) -> Ref<Vec<GlobalRef>> {
        self.globals.borrow()
    }

    fn insert_export<N: Into<String>>(&self, name: N, extern_val: ExternVal) {
        self.exports.borrow_mut().insert(name.into(), extern_val);
    }

    fn alloc_module<'i, I: Iterator<Item = &'i ExternVal>>(
        loaded_module: &Module,
        extern_vals: I,
    ) -> Result<ModuleRef, Error> {
        let module = loaded_module.module();
        let instance = ModuleRef(Rc::new(ModuleInstance::default()));

        for &Type::Function(ref ty) in module.type_section().map(|ts| ts.types()).unwrap_or(&[]) {
            let signature = Rc::new(Signature::from_elements(ty));
            instance.push_signature(signature);
        }

        {
            let mut imports = module
                .import_section()
                .map(|is| is.entries())
                .unwrap_or(&[])
                .into_iter();
            let mut extern_vals = extern_vals;
            loop {
                // Iterate on imports and extern_vals in lockstep, a-la `Iterator:zip`.
                // We can't use `Iterator::zip` since we want to check if lengths of both iterators are same and
                // `Iterator::zip` just returns `None` if either of iterators return `None`.
                let (import, extern_val) = match (imports.next(), extern_vals.next()) {
                    (Some(import), Some(extern_val)) => (import, extern_val),
                    (None, None) => break,
                    (Some(_), None) | (None, Some(_)) => {
                        return Err(Error::Instantiation(
                            "extern_vals length is not equal to import section entries".to_owned(),
                        ));
                    }
                };

                match (import.external(), extern_val) {
                    (&External::Function(fn_type_idx), &ExternVal::Func(ref func)) => {
                        let expected_fn_type = instance
                            .signature_by_index(fn_type_idx)
                            .expect("Due to validation function type should exists");
                        let actual_fn_type = func.signature();
                        if &*expected_fn_type != actual_fn_type {
                            return Err(Error::Instantiation(format!(
								"Expected function with type {:?}, but actual type is {:?} for entry {}",
								expected_fn_type,
								actual_fn_type,
								import.field(),
							)));
                        }
                        instance.push_func(func.clone())
                    }
                    (&External::Table(ref tt), &ExternVal::Table(ref table)) => {
                        match_limits(table.limits(), tt.limits())?;
                        instance.push_table(table.clone());
                    }
                    (&External::Memory(ref mt), &ExternVal::Memory(ref memory)) => {
                        match_limits(memory.limits(), mt.limits())?;
                        instance.push_memory(memory.clone());
                    }
                    (&External::Global(ref gl), &ExternVal::Global(ref global)) => {
                        if gl.content_type() != global.elements_value_type() {
                            return Err(Error::Instantiation(format!(
                                "Expect global with {:?} type, but provided global with {:?} type",
                                gl.content_type(),
                                global.value_type(),
                            )));
                        }
                        instance.push_global(global.clone());
                    }
                    (expected_import, actual_extern_val) => {
                        return Err(Error::Instantiation(format!(
                            "Expected {:?} type, but provided {:?} extern_val",
                            expected_import, actual_extern_val
                        )));
                    }
                }
            }
        }

        let code = loaded_module.code();
        {
            let funcs = module
                .function_section()
                .map(|fs| fs.entries())
                .unwrap_or(&[]);
            let bodies = module.code_section().map(|cs| cs.bodies()).unwrap_or(&[]);
            debug_assert!(
                funcs.len() == bodies.len(),
                "Due to validation func and body counts must match"
            );

            for (index, (ty, body)) in
                Iterator::zip(funcs.into_iter(), bodies.into_iter()).enumerate()
            {
                let signature = instance
                    .signature_by_index(ty.type_ref())
                    .expect("Due to validation type should exists");
                let code = code.get(index).expect(
					"At func validation time labels are collected; Collected labels are added by index; qed",
				).clone();
                let func_body = FuncBody {
                    locals: body.locals().to_vec(),
                    code: code,
                };
                let func_instance =
                    FuncInstance::alloc_internal(Rc::downgrade(&instance.0), signature, func_body);
                instance.push_func(func_instance);
            }
        }

        for table_type in module.table_section().map(|ts| ts.entries()).unwrap_or(&[]) {
            let table =
                TableInstance::alloc(table_type.limits().initial(), table_type.limits().maximum())?;
            instance.push_table(table);
        }

        for memory_type in module
            .memory_section()
            .map(|ms| ms.entries())
            .unwrap_or(&[])
        {
            let initial: Pages = Pages(memory_type.limits().initial() as usize);
            let maximum: Option<Pages> = memory_type.limits().maximum().map(|m| Pages(m as usize));

            let memory = MemoryInstance::alloc(initial, maximum)
                .expect("Due to validation `initial` and `maximum` should be valid");
            instance.push_memory(memory);
        }

        for global_entry in module
            .global_section()
            .map(|gs| gs.entries())
            .unwrap_or(&[])
        {
            let init_val = eval_init_expr(global_entry.init_expr(), &*instance);
            let global = GlobalInstance::alloc(init_val, global_entry.global_type().is_mutable());
            instance.push_global(global);
        }

        for export in module
            .export_section()
            .map(|es| es.entries())
            .unwrap_or(&[])
        {
            let field = export.field();
            let extern_val: ExternVal = match *export.internal() {
                Internal::Function(idx) => {
                    let func = instance
                        .func_by_index(idx)
                        .expect("Due to validation func should exists");
                    ExternVal::Func(func)
                }
                Internal::Global(idx) => {
                    let global = instance
                        .global_by_index(idx)
                        .expect("Due to validation global should exists");
                    ExternVal::Global(global)
                }
                Internal::Memory(idx) => {
                    let memory = instance
                        .memory_by_index(idx)
                        .expect("Due to validation memory should exists");
                    ExternVal::Memory(memory)
                }
                Internal::Table(idx) => {
                    let table = instance
                        .table_by_index(idx)
                        .expect("Due to validation table should exists");
                    ExternVal::Table(table)
                }
            };
            instance.insert_export(field, extern_val);
        }

        Ok(instance)
    }

    /// Instantiate a module with given [external values][ExternVal] as imports.
    ///
    /// See [new] for details.
    ///
    /// [new]: #method.new
    /// [ExternVal]: https://webassembly.github.io/spec/core/exec/runtime.html#syntax-externval
    pub fn with_externvals<'a, 'i, I: Iterator<Item = &'i ExternVal>>(
        loaded_module: &'a Module,
        extern_vals: I,
    ) -> Result<NotStartedModuleRef<'a>, Error> {
        let module = loaded_module.module();

        let module_ref = ModuleInstance::alloc_module(loaded_module, extern_vals)?;

        for element_segment in module
            .elements_section()
            .map(|es| es.entries())
            .unwrap_or(&[])
        {
            let offset = element_segment
                .offset()
                .as_ref()
                .expect("passive segments are rejected due to validation");
            let offset_val = match eval_init_expr(offset, &module_ref) {
                RuntimeValue::I32(v) => v as u32,
                _ => panic!("Due to validation elem segment offset should evaluate to i32"),
            };

            let table_inst = module_ref
                .table_by_index(DEFAULT_TABLE_INDEX)
                .expect("Due to validation default table should exists");

            // This check is not only for bailing out early, but also to check the case when
            // segment consist of 0 members.
            if offset_val as u64 + element_segment.members().len() as u64
                > table_inst.current_size() as u64
            {
                return Err(Error::Instantiation(
                    "elements segment does not fit".to_string(),
                ));
            }

            for (j, func_idx) in element_segment.members().into_iter().enumerate() {
                let func = module_ref
                    .func_by_index(*func_idx)
                    .expect("Due to validation funcs from element segments should exists");

                table_inst.set(offset_val + j as u32, Some(func))?;
            }
        }

        for data_segment in module.data_section().map(|ds| ds.entries()).unwrap_or(&[]) {
            let offset = data_segment
                .offset()
                .as_ref()
                .expect("passive segments are rejected due to validation");
            let offset_val = match eval_init_expr(offset, &module_ref) {
                RuntimeValue::I32(v) => v as u32,
                _ => panic!("Due to validation data segment offset should evaluate to i32"),
            };

            let memory_inst = module_ref
                .memory_by_index(DEFAULT_MEMORY_INDEX)
                .expect("Due to validation default memory should exists");
            memory_inst.set(offset_val, data_segment.value())?;
        }

        Ok(NotStartedModuleRef {
            loaded_module,
            instance: module_ref,
        })
    }

    /// Instantiate a [module][`Module`].
    ///
    /// Note that in case of successful instantiation this function returns a reference to
    /// a module which `start` function is not called.
    /// In order to complete instantiatiation `start` function must be called. However, there are
    /// situations where you might need to do additional setup before calling `start` function.
    /// For such sitations this separation might be useful.
    ///
    /// See [`NotStartedModuleRef`] for details.
    ///
    /// # Errors
    ///
    /// Returns `Err` if the module cannot be instantiated.
    ///
    /// This can happen if one of the imports can't
    /// be satisfied (e.g module isn't registered in `imports` [resolver][`ImportResolver`]) or
    /// there is a mismatch between requested import and provided (e.g. module requested memory with no
    /// maximum size limit, however, was provided memory with the maximum size limit).
    ///
    /// # Examples
    ///
    /// ```rust
    /// use wasmi::{ModuleInstance, ImportsBuilder, NopExternals};
    /// # fn func() -> Result<(), ::wasmi::Error> {
    /// # let module = wasmi::Module::from_buffer(&[0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00]).unwrap();
    ///
    /// // ModuleInstance::new returns instance which `start` function isn't called.
    /// let not_started = ModuleInstance::new(
    ///     &module,
    ///     &ImportsBuilder::default()
    /// )?;
    /// // Call `start` function if any.
    /// let instance = not_started.run_start(&mut NopExternals)?;
    ///
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// If you sure that the module doesn't have `start` function you can use [`assert_no_start`] to get
    /// instantiated module without calling `start` function.
    ///
    /// ```rust
    /// use wasmi::{ModuleInstance, ImportsBuilder, NopExternals};
    /// # fn func() -> Result<(), ::wasmi::Error> {
    /// # let module = wasmi::Module::from_buffer(&[0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00]).unwrap();
    ///
    /// // This will panic if the module actually contain `start` function.
    /// let not_started = ModuleInstance::new(
    ///     &module,
    ///     &ImportsBuilder::default()
    /// )?.assert_no_start();
    ///
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// [`Module`]: struct.Module.html
    /// [`NotStartedModuleRef`]: struct.NotStartedModuleRef.html
    /// [`ImportResolver`]: trait.ImportResolver.html
    /// [`assert_no_start`]: struct.NotStartedModuleRef.html#method.assert_no_start
    pub fn new<'m, I: ImportResolver>(
        loaded_module: &'m Module,
        imports: &I,
    ) -> Result<NotStartedModuleRef<'m>, Error> {
        let module = loaded_module.module();

        let mut extern_vals = Vec::new();
        for import_entry in module.import_section().map(|s| s.entries()).unwrap_or(&[]) {
            let module_name = import_entry.module();
            let field_name = import_entry.field();
            let extern_val = match *import_entry.external() {
                External::Function(fn_ty_idx) => {
                    let types = module.type_section().map(|s| s.types()).unwrap_or(&[]);
                    let &Type::Function(ref func_type) = types
                        .get(fn_ty_idx as usize)
                        .expect("Due to validation functions should have valid types");
                    let signature = Signature::from_elements(func_type);
                    let func = imports.resolve_func(module_name, field_name, &signature)?;
                    ExternVal::Func(func)
                }
                External::Table(ref table_type) => {
                    let table_descriptor = TableDescriptor::from_elements(table_type);
                    let table =
                        imports.resolve_table(module_name, field_name, &table_descriptor)?;
                    ExternVal::Table(table)
                }
                External::Memory(ref memory_type) => {
                    let memory_descriptor = MemoryDescriptor::from_elements(memory_type);
                    let memory =
                        imports.resolve_memory(module_name, field_name, &memory_descriptor)?;
                    ExternVal::Memory(memory)
                }
                External::Global(ref global_type) => {
                    let global_descriptor = GlobalDescriptor::from_elements(global_type);
                    let global =
                        imports.resolve_global(module_name, field_name, &global_descriptor)?;
                    ExternVal::Global(global)
                }
            };
            extern_vals.push(extern_val);
        }

        Self::with_externvals(loaded_module, extern_vals.iter())
    }

    /// Invoke exported function by a name.
    ///
    /// This function finds exported function by a name, and calls it with provided arguments and
    /// external state.
    ///
    /// # Errors
    ///
    /// Returns `Err` if:
    ///
    /// - there are no export with a given name or this export is not a function,
    /// - given arguments doesn't match to function signature,
    /// - trap occurred at the execution time,
    ///
    /// # Examples
    ///
    /// Invoke a function that takes two numbers and returns sum of them.
    ///
    /// ```rust
    /// # extern crate wasmi;
    /// # extern crate wabt;
    /// # use wasmi::{ModuleInstance, ImportsBuilder, NopExternals, RuntimeValue};
    /// # fn main() {
    /// # let wasm_binary: Vec<u8> = wabt::wat2wasm(
    /// #   r#"
    /// #   (module
    /// #       (func (export "add") (param i32 i32) (result i32)
    /// #           get_local 0
    /// #           get_local 1
    /// #           i32.add
    /// #       )
    /// #   )
    /// #   "#,
    /// # ).expect("failed to parse wat");
    /// # let module = wasmi::Module::from_buffer(&wasm_binary).expect("failed to load wasm");
    /// # let instance = ModuleInstance::new(
    /// # &module,
    /// # &ImportsBuilder::default()
    /// # ).expect("failed to instantiate wasm module").assert_no_start();
    /// assert_eq!(
    ///     instance.invoke_export(
    ///         "add",
    ///         &[RuntimeValue::I32(5), RuntimeValue::I32(3)],
    ///         &mut NopExternals,
    ///     ).expect("failed to execute export"),
    ///     Some(RuntimeValue::I32(8)),
    /// );
    /// # }
    /// ```
    pub fn invoke_export<E: Externals>(
        &self,
        func_name: &str,
        args: &[RuntimeValue],
        externals: &mut E,
    ) -> Result<Option<RuntimeValue>, Error> {
        let func_instance = self.func_by_name(func_name)?;

        FuncInstance::invoke(&func_instance, args, externals).map_err(|t| Error::Trap(t))
    }

    /// Invoke exported function by a name asynchronously.
    ///
    /// This function finds exported function by a name, and calls it with provided arguments and
    /// external state.
    ///
    /// # Errors
    ///
    /// Returns `Err` if:
    ///
    /// - there are no export with a given name or this export is not a function,
    /// - given arguments doesn't match to function signature,
    /// - trap occurred at the execution time,
    ///
    pub fn invoke_export_async<'a,E>(
        &self,
        func_name: &str,
        args: Vec<RuntimeValue>,
        externals: Rc<E>,
    ) -> impl Future<Item=Option<RuntimeValue>,Error= Error>
    where E:AsyncExternals +'static 
    {
        let func_instance = self.func_by_name(func_name);

        match func_instance{
            Ok(func)=>{
                future::Either::A(FuncInstance::invoke_async(func, args, externals).from_err())

            },
            Err(err)=>{
                future::Either::B(future::err(err))
            }
        }

        
    }

    /// Invoke exported function by a name using recycled stacks.
    ///
    /// # Errors
    ///
    /// Same as [`invoke_export`].
    ///
    /// [`invoke_export`]: #method.invoke_export
    pub fn invoke_export_with_stack<E: Externals>(
        &self,
        func_name: &str,
        args: &[RuntimeValue],
        externals: &mut E,
        stack_recycler: &mut StackRecycler,
    ) -> Result<Option<RuntimeValue>, Error> {
        let func_instance = self.func_by_name(func_name)?;

        FuncInstance::invoke_with_stack(&func_instance, args, externals, stack_recycler)
            .map_err(|t| Error::Trap(t))
    }

    fn func_by_name(&self, func_name: &str) -> Result<FuncRef, Error> {
        let extern_val = self
            .export_by_name(func_name)
            .ok_or_else(|| Error::Function(format!("Module doesn't have export {}", func_name)))?;

        match extern_val {
            ExternVal::Func(func_instance) => Ok(func_instance),
            unexpected => Err(Error::Function(format!(
                "Export {} is not a function, but {:?}",
                func_name, unexpected
            ))),
        }
    }

    /// Find export by a name.
    ///
    /// Returns `None` if there is no export with such name.
    pub fn export_by_name(&self, name: &str) -> Option<ExternVal> {
        self.exports.borrow().get(name).cloned()
    }
}

/// Mostly instantiated [`ModuleRef`].
///
/// At this point memory segments and tables are copied. However, `start` function (if any) is not called.
/// To get [fully instantiated module instance][`ModuleRef`], [running `start` function][`run_start`] is required.
///
/// You can still access not fully initialized instance by calling [`not_started_instance`],
/// but keep in mind, that this is sort of escape hatch: module really might depend on initialization
/// done in `start` function. It's definitely not recommended to call any exports on [`ModuleRef`]
/// returned by this function.
///
/// If you sure, that there is no `start` function (e.g. because you created it without one), you can
/// call [`assert_no_start`] which returns [`ModuleRef`] without calling `start` function. However,
/// it will panic if module contains `start` function.
///
/// [`ModuleRef`]: struct.ModuleRef.html
/// [`run_start`]: #method.run_start
/// [`assert_no_start`]: #method.assert_no_start
/// [`not_started_instance`]: #method.not_started_instance
pub struct NotStartedModuleRef<'a> {
    loaded_module: &'a Module,
    instance: ModuleRef,
}

impl<'a> NotStartedModuleRef<'a> {
    /// Returns not fully initialized instance.
    ///
    /// To fully initialize the instance you need to call either [`run_start`] or
    /// [`assert_no_start`]. See struct documentation for details.
    ///
    /// [`NotStartedModuleRef`]: struct.NotStartedModuleRef.html
    /// [`ModuleRef`]: struct.ModuleRef.html
    /// [`run_start`]: #method.run_start
    /// [`assert_no_start`]: #method.assert_no_start
    pub fn not_started_instance(&self) -> &ModuleRef {
        &self.instance
    }

    /// Executes `start` function (if any) and returns fully instantiated module.
    ///
    /// # Errors
    ///
    /// Returns `Err` if start function traps.
    pub fn run_start<E: Externals>(self, state: &mut E) -> Result<ModuleRef, Trap> {
        if let Some(start_fn_idx) = self.loaded_module.module().start_section() {
            let start_func = self
                .instance
                .func_by_index(start_fn_idx)
                .expect("Due to validation start function should exists");
            FuncInstance::invoke(&start_func, &[], state)?;
        }
        Ok(self.instance)
    }

    /// Executes `start` function (if any) for an asynchronous module and returns fully instantiated module.
    /// Blocks until start function has returned
    /// 
    /// # Errors
    ///
    /// Returns `Err` if start function traps.
    pub fn run_start_async_block<E:AsyncExternals +'a>(self, state: &Rc<E>) -> Result<ModuleRef, Trap> {
        if let Some(start_fn_idx) = self.loaded_module.module().start_section() {
            let start_func = self
                .instance
                .func_by_index(start_fn_idx)
                .expect("Due to validation start function should exists");
            FuncInstance::invoke_async(start_func, Vec::new(), Rc::clone(state)).wait()?;
        }
        Ok(self.instance)
    }

    /// Returns fully instantiated module without running `start` function.
    ///
    /// # Panics
    ///
    /// This function panics if original module contains `start` function.
    pub fn assert_no_start(self) -> ModuleRef {
        if self.loaded_module.module().start_section().is_some() {
            panic!("assert_no_start called on module with `start` function");
        }
        self.instance
    }

    /// Whether or not the module has a `start` function.
    ///
    /// Returns `true` if it has a `start` function.
    pub fn has_start(&self) -> bool {
        self.loaded_module.module().start_section().is_some()
    }
}

fn eval_init_expr(init_expr: &InitExpr, module: &ModuleInstance) -> RuntimeValue {
    let code = init_expr.code();
    debug_assert!(
        code.len() == 2,
        "Due to validation `code`.len() should be 2"
    );
    match code[0] {
        Instruction::I32Const(v) => v.into(),
        Instruction::I64Const(v) => v.into(),
        Instruction::F32Const(v) => RuntimeValue::decode_f32(v),
        Instruction::F64Const(v) => RuntimeValue::decode_f64(v),
        Instruction::GetGlobal(idx) => {
            let global = module
                .global_by_index(idx)
                .expect("Due to validation global should exists in module");
            global.get()
        }
        _ => panic!("Due to validation init should be a const expr"),
    }
}

fn match_limits(l1: &ResizableLimits, l2: &ResizableLimits) -> Result<(), Error> {
    if l1.initial() < l2.initial() {
        return Err(Error::Instantiation(format!(
            "trying to import with limits l1.initial={} and l2.initial={}",
            l1.initial(),
            l2.initial()
        )));
    }

    match (l1.maximum(), l2.maximum()) {
        (_, None) => (),
        (Some(m1), Some(m2)) if m1 <= m2 => (),
        _ => {
            return Err(Error::Instantiation(format!(
                "trying to import with limits l1.max={:?} and l2.max={:?}",
                l1.maximum(),
                l2.maximum()
            )));
        }
    }

    Ok(())
}

pub fn check_limits(limits: &ResizableLimits) -> Result<(), Error> {
    if let Some(maximum) = limits.maximum() {
        if maximum < limits.initial() {
            return Err(Error::Instantiation(format!(
                "maximum limit {} is less than minimum {}",
                maximum,
                limits.initial()
            )));
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{ExternVal, ModuleInstance};
    use func::FuncInstance;
    use imports::ImportsBuilder;
    use tests::parse_wat;
    use types::{Signature, ValueType};

    #[should_panic]
    #[test]
    fn assert_no_start_panics_on_module_with_start() {
        let module_with_start = parse_wat(
            r#"
			(module
				(func $f)
				(start $f))
			"#,
        );
        let module = ModuleInstance::new(&module_with_start, &ImportsBuilder::default()).unwrap();
        assert!(!module.has_start());
        module.assert_no_start();
    }

    #[test]
    fn imports_provided_by_externvals() {
        let module_with_single_import = parse_wat(
            r#"
			(module
				(import "foo" "bar" (func))
				)
			"#,
        );

        assert!(ModuleInstance::with_externvals(
            &module_with_single_import,
            [ExternVal::Func(FuncInstance::alloc_host(
                Signature::new(&[][..], None),
                0
            ),)]
            .iter(),
        )
        .is_ok());

        // externval vector is longer than import count.
        assert!(ModuleInstance::with_externvals(
            &module_with_single_import,
            [
                ExternVal::Func(FuncInstance::alloc_host(Signature::new(&[][..], None), 0)),
                ExternVal::Func(FuncInstance::alloc_host(Signature::new(&[][..], None), 1)),
            ]
            .iter(),
        )
        .is_err());

        // externval vector is shorter than import count.
        assert!(ModuleInstance::with_externvals(&module_with_single_import, [].iter(),).is_err());

        // externval vector has an unexpected type.
        assert!(ModuleInstance::with_externvals(
            &module_with_single_import,
            [ExternVal::Func(FuncInstance::alloc_host(
                Signature::new(&[][..], Some(ValueType::I32)),
                0
            ),)]
            .iter(),
        )
        .is_err());
    }
}
