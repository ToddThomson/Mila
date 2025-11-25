# Mila Deep Learning Framework - Beta Release Roadmap

**Target:** v0.1.0-beta  
**Timeline:** 10 weeks  
**Estimated Completion:** Mid-February 2025

---

## Current State

### ✅ Completed
- Core NN components with backward() implemented (Linear, Gelu, Softmax, CrossEntropy, etc.)
- CUDA + CPU kernels for all built-in components and backend operations
- MLP Block implementation
- MNIST network implementation
- MNIST data pipeline (functional)
- DatasetReader interface (good, not production-ready)
- AdamW optimizer (working with MNIST training loop)
- Low-level training API (working)
- Test-driven development (~60% coverage)

### 🚧 Needs Work
- Network-level API (works but needs refinement)
- Model class (experimental, unstable API, minimal functionality)
- Loss functions on GPU (currently CPU-only with CPU gradients)
- Testing completion (40% remaining)

### ❌ Out of Scope
- Autograd system (manual backward() only - design decision)

---

## Beta Success Criteria

### Technical Requirements
- [ ] MNIST accuracy ≥ 98% (MLP)
- [ ] Training works on both CPU and CUDA
- [ ] Checkpointing works reliably
- [ ] No memory leaks over 100 epochs
- [ ] Loss functions run on GPU
- [ ] Test coverage ≥ 90%

### User Experience Requirements
- [ ] Complete MNIST example with instructions
- [ ] Getting started documentation
- [ ] API reference for core classes
- [ ] README with clear value proposition

### Code Quality Requirements
- [ ] All tests passing in CI
- [ ] No critical bugs
- [ ] Stable API committed to backward compatibility

---

## Phase 1: Model API Design & Loss Migration (Weeks 1-2)

**Goal:** Stabilize Model class API and move loss to GPU.

### Week 1: Model Class API Finalization

**Tasks:**
- [ ] Finalize Model constructor signature
```cpp
  Model(
      std::unique_ptr<Network<TDevice>> network,
      std::unique_ptr<Optimizer<TDevice, TPrecision>> optimizer,
      std::unique_ptr<Loss<TDevice, TPrecision>> loss,
      ModelConfig config = ModelConfig::defaults()
  );
```
- [ ] Implement Model::train() method
```cpp
  template<typename TReader>
  TrainingHistory train(TReader& train_reader, TReader* val_reader = nullptr);
```
- [ ] Implement Model::trainEpoch() - wire to existing low-level loop
- [ ] Implement Model::validateEpoch()
- [ ] Test Model.train() with MNIST using low-level components
- [ ] Verify loss decreases and reaches >95% accuracy
- [ ] Complete Module → Component rename across codebase
  - [ ] Update all class declarations
  - [ ] Update all inheritance statements
  - [ ] Update all documentation
  - [ ] Update all comments

**Deliverable:** Model.train(mnist_reader) works end-to-end on CPU.

### Week 2: Loss Function Migration to GPU

**Tasks:**
- [ ] Move CrossEntropyLoss from CPU to GPU
  - [ ] Implement CUDA kernel for cross_entropy_forward
  - [ ] Implement CUDA kernel for cross_entropy_backward
  - [ ] Add softmax fusion optimization
- [ ] Make Loss inherit from Component
```cpp
  template<DeviceType TDevice, TensorDataType TPrecision>
  class Loss : public Component<TDevice> {
      virtual Tensor compute(const Tensor& prediction, const Tensor& target) = 0;
      virtual void optimizeNetworkGraph(Network<TDevice>& network) {}
  };
```
- [ ] Implement CrossEntropyLoss with network optimization hook
  - [ ] Detect and remove final softmax layer if present
  - [ ] Use fused softmax_cross_entropy kernel
- [ ] Implement MSELoss (simpler, good for testing)
- [ ] Test loss backward() correctness with numerical gradients
- [ ] Wire into Model::train() and verify GPU training works

**Deliverable:** Loss runs on GPU, integrated with Model, MNIST training works on CUDA.

---

## Phase 2: Model Features & Checkpointing (Weeks 3-4)

**Goal:** Complete Model functionality.

### Week 3: Model Core Features

**Tasks:**
- [ ] Implement Model::evaluate()
```cpp
  template<typename TReader>
  double evaluate(TReader& test_reader);
```
- [ ] Implement Model::resumeTraining()
```cpp
  TrainingHistory resumeTraining(size_t additional_epochs);
```
- [ ] Add device transfer logic in training loop
```cpp
  auto device_inputs = transferToDevice(reader.inputs());
```
- [ ] Add precision conversion if needed
```cpp
  auto converted = convertPrecision<TPrecision>(device_inputs);
```
- [ ] Implement TrainingHistory tracking
  - [ ] Per-epoch train loss
  - [ ] Per-epoch val loss
  - [ ] Best val loss tracking
  - [ ] Epochs without improvement (for early stopping)
- [ ] Add early stopping logic based on config
- [ ] Add verbose logging with epoch progress
```
  Epoch 1/100: loss = 0.5234, val_loss = 0.4821
  Epoch 2/100: loss = 0.3456, val_loss = 0.3912
```

**Deliverable:** Model has all core training features working.

### Week 4: Checkpointing Implementation

**Tasks:**
- [ ] Implement Model::saveCheckpoint()
```cpp
  void saveCheckpoint(const std::string& filepath) const;
```
  - [ ] Save network state (all component parameters)
  - [ ] Save optimizer state (momentum, velocity buffers)
  - [ ] Save loss state (if any learnable parameters)
  - [ ] Save config
  - [ ] Save metadata (epoch, train_loss, val_loss, timestamp)
- [ ] Implement Model::fromCheckpoint()
```cpp
  static std::unique_ptr<Model> fromCheckpoint(
      const std::string& filepath,
      std::shared_ptr<ExecutionContext<TDevice>> ctx
  );
```
- [ ] Implement Model::exportForInference()
```cpp
  void exportForInference(const std::string& filepath) const;
```
  - [ ] Save only network weights (no optimizer state)
  - [ ] Mark as export_mode in metadata
- [ ] Implement Model::loadForInference()
```cpp
  static std::unique_ptr<Network<TDevice>> loadForInference(
      const std::string& filepath,
      std::shared_ptr<ExecutionContext<TDevice>> ctx
  );
```
- [ ] Test checkpoint save/load cycle
- [ ] Test training resume from checkpoint
- [ ] Test cross-device loading (save on CUDA, load on CPU)
- [ ] Test checkpoint versioning and compatibility

**Deliverable:** Full checkpoint lifecycle works reliably.

---

## Phase 3: Network API Polish (Week 5)

**Goal:** Stabilize Network-level API based on Model usage patterns.

### Week 5: Network Refinement

**Tasks:**
- [ ] Review Network API based on Model integration learnings
- [ ] Ensure Network properly implements Component interface
```cpp
  void setMode(ComputationMode mode);
  std::vector<Parameter*> parameters();
  void save(Archive& archive, SerializationMode mode);
  void load(Archive& archive, SerializationMode mode);
```
- [ ] Ensure Network properly manages component lifecycle
  - [ ] Component ownership
  - [ ] Parameter aggregation from child components
  - [ ] Mode propagation to all child components
- [ ] Test Network serialization/deserialization thoroughly
  - [ ] Save network with all components
  - [ ] Load and verify parameter values
  - [ ] Test with nested networks
- [ ] Add Network::summary() for debugging
```cpp
  std::string summary() const;  // Print architecture, param counts
```
- [ ] Polish any rough edges discovered during Model integration
- [ ] Add helper methods if needed by Model

**Deliverable:** Network API is stable and well-integrated with Model.

---

## Phase 4: Testing Completion (Weeks 6-7)

**Goal:** Reach 90%+ test coverage and ensure robustness.

### Week 6: Model & Integration Tests

**Tasks:**
- [ ] Write unit tests for Model class
  - [ ] Constructor with various configurations
  - [ ] train() with different configs (epochs, batch_size, etc.)
  - [ ] train() with validation data
  - [ ] evaluate() correctness
  - [ ] resumeTraining() from various epochs
  - [ ] Checkpoint save/load cycle
  - [ ] Export/load for inference
- [ ] Write integration tests
  - [ ] End-to-end MNIST training (CPU)
  - [ ] End-to-end MNIST training (CUDA)
  - [ ] Train → save → load → resume → train more
  - [ ] Train → export → load for inference → evaluate
  - [ ] Mixed precision training (FP32 vs FP16)
- [ ] Test edge cases
  - [ ] Empty dataset (should throw)
  - [ ] Single sample dataset
  - [ ] Very small batches (batch_size=1)
  - [ ] Very large batches
  - [ ] Different input shapes
  - [ ] Mismatched network/data dimensions
- [ ] Memory leak detection
  - [ ] Run training for 1000 epochs, monitor memory usage
  - [ ] Check for tensor leaks
  - [ ] Check for parameter leaks
- [ ] Numerical stability tests
  - [ ] Gradient explosion scenarios
  - [ ] Very small learning rates (1e-10)
  - [ ] Very large learning rates (1.0)
  - [ ] NaN/Inf detection and handling

**Deliverable:** Model is thoroughly tested with integration tests passing.

### Week 7: Component & Loss Tests

**Tasks:**
- [ ] Complete remaining component tests (to reach 90%+ coverage)
  - [ ] Forward pass correctness for all layers
  - [ ] Backward pass correctness (numerical gradient checks)
  - [ ] Parameter initialization
  - [ ] Device transfer
  - [ ] Mode switching (training vs inference)
- [ ] Write comprehensive tests for Loss classes
  - [ ] CrossEntropyLoss forward correctness
  - [ ] CrossEntropyLoss backward correctness (numerical gradients)
  - [ ] MSELoss forward/backward correctness
  - [ ] Loss device transfer (CPU ↔ CUDA)
  - [ ] Loss serialization/deserialization
  - [ ] Network optimization hook (softmax removal)
- [ ] Test Loss as Component
  - [ ] parameters() returns empty for stateless losses
  - [ ] parameters() returns learnable params for stateful losses
  - [ ] setMode() behavior (if any)
  - [ ] save/load cycle preserves state
- [ ] Fix any bugs discovered during testing
- [ ] Achieve 90%+ code coverage overall

**Deliverable:** 90%+ test coverage achieved, all tests passing, CI green.

---

## Phase 5: Examples & Validation (Week 8)

**Goal:** Prove Mila works with complete, production-quality examples.

### Week 8: MNIST Example Refinement

**Tasks:**
- [ ] Create polished MNIST example
  - [ ] examples/mnist/train_mnist_mlp.cpp (simple MLP)
  - [ ] examples/mnist/train_mnist_cnn.cpp (LeNet-style CNN)
  - [ ] examples/mnist/inference.cpp (load and evaluate)
- [ ] Both CPU and CUDA variants
  - [ ] Command-line flag to select device
  - [ ] Graceful fallback if CUDA unavailable
- [ ] Demonstrate all Model features
  - [ ] Training from scratch
  - [ ] Checkpoint saving during training
  - [ ] Resume training from checkpoint
  - [ ] Export trained model
  - [ ] Load exported model for inference
- [ ] Achieve 98%+ test accuracy consistently
  - [ ] MLP: 97-98% accuracy
  - [ ] CNN: 98-99% accuracy
- [ ] Benchmark training time
  - [ ] Compare CPU vs CUDA performance
  - [ ] Document performance numbers in README
  - [ ] Time per epoch, total training time
- [ ] Add training visualization support
  - [ ] Output training curves to CSV
  - [ ] Optional: simple terminal progress bar
- [ ] Write comprehensive comments in example code
  - [ ] Explain each step
  - [ ] Reference API documentation
- [ ] Create example-specific README
  - [ ] How to build
  - [ ] How to run
  - [ ] Expected results

**Deliverable:** Production-quality MNIST example that serves as reference implementation.

---

## Phase 6: Transformer Example (Week 9)

**Goal:** Demonstrate Mila works for modern, complex architectures.

### Week 9: Transformer Example

**Tasks:**
- [ ] Create simple transformer example
  - [ ] Character-level language model (Shakespeare corpus)
  - [ ] Or simple sequence-to-sequence translation task
- [ ] Wire existing transformer implementation to Model.train()
- [ ] Create appropriate DatasetReader for sequence data
- [ ] Train to reasonable perplexity/accuracy
  - [ ] Perplexity < 2.0 for character-level
  - [ ] Or BLEU > 20 for translation
- [ ] Document transformer architecture
  - [ ] Layers, attention heads, dimensions
  - [ ] Total parameter count
- [ ] Add inference example
  - [ ] Text generation with temperature sampling
  - [ ] Beam search (optional)
- [ ] Create example README with results

**Deliverable:** Transformer example shows Mila handles complex, modern architectures.

**Note:** This phase is optional for beta and can be deferred to v0.2 if timeline is tight. Priority is MNIST working perfectly.

---

## Phase 7: Documentation (Week 10)

**Goal:** Make Mila accessible to external users.

### Week 10: Documentation Sprint

**Tasks:**
- [ ] **Architecture Overview Document**
  - [ ] Component hierarchy explanation
  - [ ] Network composition patterns
  - [ ] Model orchestration workflow
  - [ ] Training loop internals
  - [ ] Design philosophy (composition over inheritance)
  - [ ] Why no autograd (manual backward design)
- [ ] **API Reference**
  - [ ] Component class
    - [ ] All virtual methods
    - [ ] Lifecycle (construction, parameters, modes, serialization)
  - [ ] Network class
    - [ ] Building networks
    - [ ] Adding components
    - [ ] Parameter management
  - [ ] Model class
    - [ ] Construction
    - [ ] Training workflow
    - [ ] Checkpointing
    - [ ] Inference export
  - [ ] Optimizer interface
    - [ ] Available optimizers (AdamW, SGD if implemented)
    - [ ] Configuration options
  - [ ] Loss interface
    - [ ] Available losses (CrossEntropy, MSE)
    - [ ] Custom loss creation
  - [ ] DatasetReader interface
    - [ ] Creating custom readers
    - [ ] Batching and shuffling
- [ ] **Tutorials**
  - [ ] Getting Started
    - [ ] Installation instructions
    - [ ] Building your first model
    - [ ] Training on sample data
  - [ ] Building a Network
    - [ ] Layer-by-layer construction
    - [ ] Using built-in components
    - [ ] Network composition patterns
  - [ ] Training Your First Model
    - [ ] Complete MNIST walkthrough
    - [ ] Explaining each step
    - [ ] Interpreting results
  - [ ] Saving and Loading Models
    - [ ] Checkpointing during training
    - [ ] Resuming interrupted training
    - [ ] Exporting for deployment
  - [ ] Creating Custom Layers (advanced)
    - [ ] Implementing forward()
    - [ ] Implementing backward()
    - [ ] Parameter management
    - [ ] Testing your layer
- [ ] **README.md**
  - [ ] Project overview and motivation
  - [ ] Key features
    - [ ] C++23 modules
    - [ ] Manual gradient design
    - [ ] CUDA + CPU support
    - [ ] Type-safe device/precision handling
  - [ ] Quick start example
  - [ ] Installation instructions
  - [ ] Links to examples
  - [ ] Links to documentation
  - [ ] Badges (build status, test coverage, license)
  - [ ] Roadmap to v1.0
  - [ ] Contributing guidelines
- [ ] **Code Documentation**
  - [ ] Review all public APIs for documentation completeness
  - [ ] Ensure all classes have class-level docs
  - [ ] Ensure all public methods have method-level docs
  - [ ] Add code examples in docs where helpful

**Deliverable:** Comprehensive documentation enabling self-service onboarding.

---

## Phase 8: Beta Release (Week 10, End of Week)

**Goal:** Package and release v0.1.0-beta.

### Final Week: Release Preparation

**Tasks:**
- [ ] **Create Release Checklist**
  - [ ] All tests passing in CI
  - [ ] Test coverage ≥ 90%
  - [ ] MNIST MLP accuracy ≥ 98%
  - [ ] MNIST CNN accuracy ≥ 98%
  - [ ] Documentation complete
  - [ ] Examples working and documented
  - [ ] No known critical bugs
  - [ ] Memory leak tests passing
- [ ] **Version Tagging**
  - [ ] Tag commit as v0.1.0-beta
  - [ ] Create GitHub release
  - [ ] Attach pre-built binaries (optional)
- [ ] **Write Release Notes**
  - [ ] Features included
    - [ ] High-level Model API
    - [ ] Component-based architecture
    - [ ] CUDA + CPU support
    - [ ] Checkpointing system
    - [ ] MNIST examples
    - [ ] Transformer support
  - [ ] Known limitations
    - [ ] No autograd (by design)
    - [ ] Limited optimizer selection
    - [ ] DatasetReader not production-ready
    - [ ] No distributed training
  - [ ] Roadmap for v0.2
    - [ ] Additional optimizers (SGD with momentum, RMSprop)
    - [ ] More loss functions
    - [ ] BatchNorm, LayerNorm, Dropout
    - [ ] Production-ready data pipeline
    - [ ] Learning rate schedulers
    - [ ] Callbacks/hooks system
- [ ] **Set Up Issue Templates**
  - [ ] Bug report template
  - [ ] Feature request template
  - [ ] Documentation improvement template
  - [ ] Question template
- [ ] **Community Setup**
  - [ ] Enable GitHub Discussions
  - [ ] Create CONTRIBUTING.md
  - [ ] Create CODE_OF_CONDUCT.md
  - [ ] Set up Discord/Slack (optional)
- [ ] **Announce Beta**
  - [ ] GitHub Discussions post
  - [ ] Reddit r/MachineLearning
  - [ ] Reddit r/cpp
  - [ ] Twitter/X announcement
  - [ ] Hacker News "Show HN" (maybe wait for feedback first)
  - [ ] Email to interested parties
- [ ] **Prepare for Feedback**
  - [ ] Set up GitHub issue monitoring
  - [ ] Plan hotfix process
  - [ ] Allocate time for user support

**Deliverable:** 🎉 **Mila v0.1.0-beta is released and announced!**

---

## Post-Beta: v0.2 Roadmap (Not Part of Beta)

Items deferred to next release:

### Additional Components
- [ ] BatchNorm
- [ ] Dropout
- [ ] Embedding layer
- [ ] More activation functions (ReLU, SiLU, etc.)

### Training Features
- [ ] Learning rate schedulers (StepLR, CosineAnnealing, etc.)
- [ ] Callbacks/hooks system for custom training logic
- [ ] Gradient accumulation
- [ ] Mixed precision training (automatic)

### Optimizers
- [ ] SGD with momentum
- [ ] RMSprop
- [ ] AdaGrad

### Data Pipeline
- [ ] Production-ready DatasetReader
- [ ] Data augmentation support
- [ ] Multi-threaded data loading
- [ ] Prefetching

### Additional Examples
- [ ] CIFAR-10 (ResNet)
- [ ] Image segmentation
- [ ] Object detection (if time permits)

### Performance
- [ ] Kernel fusion optimizations
- [ ] Memory pool optimizations
- [ ] Multi-GPU support

---

## Risk Mitigation

### High-Risk Items (Need Early Focus)
1. **Loss GPU migration** - Critical path, potential for subtle bugs
   - Mitigation: Extensive numerical gradient testing
2. **Model API stability** - Must get this right for beta
   - Mitigation: Design review before implementation
3. **Memory management** - Leaks will damage credibility
   - Mitigation: Dedicated leak testing in Week 6
4. **Numerical stability** - Gradients can be finicky
   - Mitigation: Test with extreme learning rates, long training runs

### Medium-Risk Items
1. **Checkpointing** - Versioning and compatibility issues
   - Mitigation: Thorough testing of save/load cycles
2. **Device transfers** - Potential for subtle bugs
   - Mitigation: Cross-device testing (CPU ↔ CUDA)
3. **Documentation completeness** - Easy to underestimate time needed
   - Mitigation: Allocate full week, start earlier if possible

### Low-Risk Items
1. **Examples** - Can iterate based on core functionality
2. **Network polish** - Low-level API already works
3. **Testing** - Time-consuming but straightforward

---

## Progress Tracking

Checklist tracking weekly progress:

- [ ] Week 1: Model API finalized, Module→Component rename complete
- [ ] Week 2: Loss on GPU, inheritance from Component working
- [ ] Week 3: Model features complete (train, evaluate, resume)
- [ ] Week 4: Checkpointing fully functional
- [ ] Week 5: Network API polished and stable
- [ ] Week 6: Model & integration tests complete
- [ ] Week 7: 90%+ test coverage achieved
- [ ] Week 8: MNIST example polished and achieving 98%
- [ ] Week 9: Transformer example working (optional)
- [ ] Week 10: Documentation complete, beta released
