# Tensor System Design Summary

This document tracks the major design decisions made during the tensor system redesign for the Mila Deep Neural Network library. Each decision includes the rationale and implications for the overall architecture.

## Alignment with Project Guidelines

This tensor design directly supports the project's stated priorities from [copilot-instructions.md](.github/copilot-instructions.md):

- **Addresses the highest priority development task**: Tensor system redesign
- **Implements all five tensor component goals**: API consistency, memory management, type safety, performance, and testing
- **Provides architectural foundation**: For stable 1.0 beta API development
- **Supports CPU/CUDA equivalence testing**: Through template-based architecture
- **Follows modern C++23 standards**: Using RAII principles and zero-cost abstractions


## 1. Template-Based Architecture with Memory Resource Abstraction

**Decision**: Use template specialization with `Tensor<TDataType, TMemoryResource>` pattern
**Rationale**: 
- Enables compile-time type safety and device compatibility validation
- Allows zero-cost abstractions for different memory types (Host, CUDA, Pinned, Managed)
- Supports efficient device-specific optimizations without runtime overhead
- Facilitates CPU/CUDA code equivalence testing

**Implications**:
- Compile-time validation of device compatibility
- Reduced runtime type checking overhead
- Clear separation between host and device accessible memory
- Type-safe device transfers with explicit memory resource conversion

## 2. Memory Resource Strategy

**Decision**: Abstract memory management through specialized resource types
**Current Memory Resources**:
- `HostMemoryResource` - CPU-accessible memory
- `CudaMemoryResource` - GPU-only memory (device accessible only)
- `CudaPinnedMemoryResource` - Page-locked host memory (both host and device accessible)
- `CudaManagedMemoryResource` - Unified memory (both accessible with automatic migration)

**Rationale**:
- Explicit control over memory placement and accessibility
- Optimization for specific use cases (e.g., pinned memory for efficient transfers)
- Clear performance implications through type system
- Prevention of host access to device-only memory at compile time

**Implications**:
- Memory access patterns are enforced by the type system
- Device transfers require explicit resource type conversion
- Performance characteristics are visible in the API
- Memory resource selection guides optimal data pipeline design

## 3. Abstract Data Type System with Device-Specific Trait Backends

**Decision**: Transition from concrete type template parameter to abstract `TensorDataType` enumeration with device-specific trait implementations

**Core Problem Solved**: 
The previous concrete type system (`Tensor<half, MemoryResource>`) created compilation boundary violations where device-specific types leaked into host-only code, forcing unnecessary compilation dependencies and preventing clean architectural separation between core tensor logic and device-specific implementations.

**Rationale**:
- **Host Compilation Safety**: Eliminates compilation errors when device-only types (e.g., `half`, `nv_bfloat16`, `__nv_fp8_e4m3`) are referenced in host-only compilation units
- **Architectural Purity**: Enforces clean separation between core tensor logic and device-specific implementations, moving all device code to compute backends (`dnn/compute/devices/cpu` and `dnn/compute/devices/cuda`)
- **Backend Modularity**: Enables optional compilation of device backends without affecting core tensor functionality - essential for deployment flexibility
- **Device Independence**: Allows writing generic tensor code that works across different devices without device-specific type dependencies
- **Expandable Type System**: New data types can be added to device backends without modifying core tensor implementation
- **Mixed-Precision Workflows**: Simplified handling of diverse precision types in data pipelines and model layers
- **Clean Separation of Concerns**: Device capabilities are encapsulated in device-specific trait modules

**Implementation Strategy**:
- Core tensor uses abstract enumeration - no device dependencies template<TensorDataType TDataType, typename TMemoryResource> class Tensor { TensorDataType getDataType() const override { return TDataType; } // No concrete device types in core implementation };
- Device backends provide concrete type mappings in isolated compilation units // CudaTensorTraits.ixx (CUDA compilation context only) template <> struct TensorTrait<half> { static constexpr TensorDataType data_type = TensorDataType::FP16; static constexpr bool is_device_only = true; };
- CpuTensorTraits.ixx (Host compilation context only) class CpuTensorTraits { template<TensorDataType TDataType> static consteval bool supports() { return TDataType == TensorDataTypeFP32 || TDataType == TensorDataTypeINT32; // No FP16 on CPU } };
- Runtime dispatch handles concrete operations when needed switch(tensor.getDataType()) { case TensorDataTypeFP16: // Dispatch to device-specific FP16 implementation break; case TensorDataTypeFP32: // Dispatch to appropriate FP32 implementation break; }

**Architectural Benefits**:
- **Compilation Independence**: Host-only code can reference any `TensorDataType` without device headers
- **Modular Build System**: CMake can conditionally include device backends based on hardware availability
- **Backend Isolation**: Complete separation of device-specific code in compute backend directories
- **API Stability**: Core tensor API becomes independent of device capability evolution
- **Testing Simplification**: Core tensor tests don't require device hardware or compilation contexts

**Implications**:
- **Compile-time validation**: Device trait systems provide type/device compatibility checking without exposing concrete types
- **Runtime dispatch required**: Operations needing concrete types must implement appropriate dispatch mechanisms
- **Backend isolation**: Device-specific code completely isolated in `dnn/compute/devices/*` directories
- **Migration path**: Gradual transition from concrete types to abstract enumeration across codebase
- **Performance maintained**: Zero-cost abstractions through compile-time dispatch and template specialization

**Migration Challenges Addressed**:
- Existing concrete type tensor code requires gradual transition to abstract types
- Runtime dispatch mechanisms needed for performance-critical operations requiring concrete types
- Template specialization patterns refactored to work with abstract enumeration
- Device trait unification between old `TensorTrait<concrete_type>` and new `TensorDataTypeTraits<TensorDataType>` systems

**Strategic Impact**:
This change represents the architectural maturation from research prototype to production library, enabling:
- Clean modular compilation suitable for production deployment
- Scalable device backend architecture that can evolve independently  
- Stable public APIs that remain consistent as new devices and data types are added
- Simplified development workflows where teams can work on device backends independently

## 4. Separation of Tensor Construction and Initialization

**Decision**: Tensor class separates memory allocation (constructors) from value initialization (helper functions)

**Rationale**:
- **Clean Constructor Interface**: Constructors focus solely on shape definition, memory allocation, and buffer setup
- **Extensible Initialization**: Easy to add new initialization algorithms without modifying core Tensor class
- **Device-Agnostic Operations**: Initialization functions leverage memory resource abstraction for cross-device compatibility
- **ML Framework Conventions**: Follows established patterns from PyTorch (`torch.zeros()`, `torch.nn.init.xavier_uniform_()`) and TensorFlow
- **Type Safety**: Proper handling of abstract data types through device trait systems
- **Performance Optimization**: Avoids unnecessary initialization when loading pre-trained weights
- **Testing Clarity**: Clear separation between allocation and value initialization issues

**Architecture**:
- **Tensor Constructors**: Handle shape definition, memory allocation, and buffer setup only
- **TensorInitializers Module**: Provides specialized initialization functions (`zeros()`, `ones()`, `random()`, `xavier()`, `fill()`, etc.)
- **TensorBuffer Infrastructure**: Retains value-initialization constructor to support initialization algorithms
- **Memory Resource Integration**: Initialization functions use memory resource abstraction for device-specific operations

**Benefits**:
- **Flexibility**: Same tensor can be re-initialized with different algorithms without reconstruction
- **Performance**: Construction overhead eliminated when values will be overwritten (e.g., loading weights)
- **Extensibility**: New initialization algorithms added without core class modification
- **Device Independence**: Initialization functions handle CPU/CUDA differences through memory resources
- **Debugging**: Separate allocation and initialization failure modes for clearer diagnostics

**Implications**:
- **TensorBuffer Design**: Value-initialization constructor remains essential for infrastructure support
- **Memory Resource Requirement**: Device-specific initialization requires memory resource abstraction
- **Device Trait Dependency**: Proper type conversion across abstract data types requires trait systems
- **API Pattern**: Consistent two-step pattern (construct then initialize) across all tensor creation

## 5. TensorBuffer Internal Implementation

**Decision**: TensorBuffer is an internal implementation class used exclusively by the Tensor class for memory management

**Current Status**: TensorBuffer is exported in the main Mila module (`Mila.ixx`) but only for unit testing purposes. This export should be removed once comprehensive Tensor testing is in place.

**Rationale**:
- **Encapsulation**: TensorBuffer handles low-level memory management details that should be hidden from users
- **API Minimization**: Users should interact only with the Tensor class, not directly with its internal memory buffer
- **Testing Requirements**: Temporary export allows for thorough unit testing of memory management functionality
- **Design Principle**: Internal implementation classes should not be part of the public API surface

**TensorBuffer API Methods Required by Tensor**:
Based on analysis of the Tensor class implementation, the following TensorBuffer methods are required:

- **Construction**: Default constructor, size constructor, size+value constructor, external memory constructor
- **Memory Access**: `rawData()` (both const and non-const variants)
- **Properties**: `size()`, `empty()`, `isAligned()`, `storageBytes()`, `alignedSize()`
- **Memory Operations**: `copyFrom()` for data transfer operations
- **Move Semantics**: Move constructor and move assignment for efficient ownership transfer

**Methods NOT Required by Tensor**:
- `resize()` - Tensor uses fixed-size buffers, resizing handled at Tensor level through buffer replacement
- Memory tracking functionality (template parameter `TrackMemory`) - Used only for debugging/profiling
- Debug/logging methods - Internal implementation details

**Implementation Guidelines**:
- TensorBuffer should expose only methods required by the Tensor class
- No additional convenience methods should be added without corresponding Tensor requirements
- All TensorBuffer API methods must have clear justification in terms of Tensor functionality
- Future changes to TensorBuffer API must be driven by Tensor requirements, not external usage

**Migration Plan**:
1. **Phase 1** (Current): TensorBuffer temporarily exported for unit testing
2. **Phase 2**: Complete Tensor unit tests to cover all TensorBuffer functionality through Tensor interface
3. **Phase 3**: Remove TensorBuffer export from main module, keeping it as internal implementation
4. **Phase 4**: Refactor TensorBuffer to expose only methods required by Tensor class

**Strategic Impact**:
- Maintains clean separation between public API (Tensor) and internal implementation (TensorBuffer)
- Enables internal optimization of memory management without affecting public API
- Allows for future memory management improvements without breaking changes
- Supports the alpha development philosophy of implementing optimal architecture without backward compatibility concerns

## Development Philosophy: Alpha Stage - No Backward Compatibility

**Core Principle**: This project is in active alpha development where API stability is not a constraint. All design decisions prioritize optimal architecture over backward compatibility.

**Implementation Guidelines**:
- **Clean Architecture Only**: Implement target designs directly without transition layers or compatibility shims
- **No Deprecated Patterns**: Remove old approaches rather than marking them deprecated
- **Breaking Changes Expected**: API evolution is part of the alpha development process - embrace it
- **Forward-Looking Code**: Focus exclusively on the production-ready 1.0 architecture
- **Single Implementation Path**: Avoid dual systems, migration helpers, or compatibility modes
- **Immediate Modernization**: Replace legacy patterns with optimal solutions immediately

**Rationale**: 
Alpha stage development provides the opportunity to implement optimal designs without legacy constraints. This approach leads to:
- Cleaner, more maintainable codebase
- Faster development velocity toward stable 1.0 release  
- Superior architectural decisions unencumbered by historical choices
- Simplified testing and validation (no dual-system complexity)
- Clear, consistent APIs from the start

**Example Applications**:
- `isValidTensor` concept: Single implementation for `TensorDataType` only (no concrete type support)
- Template parameters: Direct transition to abstract enumeration (no backward-compatible overloads)
- Memory resource validation: Pure abstract type system validation (no legacy concrete type validation)
- Device trait systems: Clean separation in compute backends (no migration bridges)

This philosophy enables the tensor system redesign to achieve its goal of providing a solid foundation for the 1.0 beta release without the complexity and maintenance burden of supporting obsolete patterns.