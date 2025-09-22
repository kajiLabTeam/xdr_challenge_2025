# xDR Challenge 2025 Technical Documentation

## Table of Contents
1. [System Overview](#system-overview)
2. [Architecture](#architecture)
3. [Positioning Technologies](#positioning-technologies)
4. [Implementation Details](#implementation-details)
5. [Performance Metrics](#performance-metrics)

## System Overview

The xDR Challenge 2025 system is a sophisticated indoor positioning solution designed for real-time, high-precision localization in challenging environments. Our approach combines multiple sensor modalities and estimation techniques to achieve robust positioning performance.

### Key Features
- Multi-sensor fusion architecture
- Real-time processing capability
- Adaptive estimation mode selection
- Comprehensive evaluation framework

## Architecture

### System Components

The system architecture follows a modular design pattern with clear separation of concerns:

1. **Main Pipeline (`src/pipeline.py`)**
   - Orchestrates the entire positioning workflow
   - Manages requester-localizer interaction loop
   - Handles sensor data processing and position estimation
   - Controls ground truth comparison and evaluation

2. **Requester Module (`src/lib/requester/`)**
   - **Requester**: Standard EvAAL API communication
   - **ImmediateRequester**: Offline mode for testing without API
   - Handles reload, state, nextdata, and estimates requests
   - Provides graceful error handling and status codes

3. **Localizer Module (`src/lib/localizer/`)**
   - Central hub integrating PDR, VIO, and UWB localizers
   - Implements intelligent multi-sensor fusion strategy
   - Dynamic estimation mode switching based on confidence
   - Maintains position history and trajectory state

4. **Data Recorder System (`src/lib/recorder/`)**
   - Eight specialized recorders for different sensor types:
     * ACCE, GYRO, MAGN (inertial sensors)
     * AHRS (attitude reference)
     * UWBP, UWBT (UWB proximity and time-based)
     * GPOS (GPS position), VISO (visual odometry)
   - Unified interface for temporal data access
   - Last appended data tracking for real-time processing

5. **Visualizer Module (`src/lib/visualizer/`)**
   - Real-time map plotting with trajectory overlay
   - Ground truth comparison visualization
   - Yaw angle time series plotting
   - Configurable map origin and pixel-per-meter scaling

### Data Flow

```
EvAAL API/Local Files → Requester → SensorData → DataRecorder → Localizer
                                                      ↓              ↓
                                          [ACCE, GYRO, MAGN,    Position
                                           AHRS, UWBP, UWBT,    Estimation
                                           GPOS, VISO]             ↓
                                                              TimedPose → API
                                                                  ↓
                                                              Visualization
```

## Positioning Technologies

### 1. Pedestrian Dead Reckoning (PDR)

PDR provides continuous position tracking using inertial sensors with advanced features:

**Methodology:**
- **Step Detection**: Peak detection on accelerometer magnitude using configurable distance and height thresholds
- **Step Length Estimation**: Dynamic stride calculation using `stride_scale * (max_acc - min_acc)^0.25`
- **Heading Determination**: Cumulative gyroscope integration with device orientation compensation
- **Dynamic Device Orientation Detection**: Automatic detection of device orientation (inverted/normal) based on gyroscope patterns

**Key Innovation - Device Orientation Estimation:**
- Analyzes 20 seconds of gyroscope data to determine device holding pattern
- Uses cumulative rotation and sectional analysis for robust detection
- Automatically adjusts angle calculations based on detected orientation
- Provides confidence levels (HIGH/MEDIUM/LOW) for orientation detection

**Implementation Details:**
- Moving average filters with configurable window sizes (`window_acc_sec`, `window_gyro_sec`)
- Peak detection using `scipy.signal.find_peaks` with distance and height parameters
- Trajectory calculation with initial pose and direction consideration
- Handles cases with insufficient gyroscope data gracefully

### 2. Visual-Inertual Odometry (VIO)

VIO combines visual odometry data with global positioning for accurate localization:

**Methodology:**
- **Global Position Integration**: Uses GPOS (GPS) data for absolute positioning reference
- **Relative Motion Tracking**: Processes VISO data for relative displacement calculation
- **Procrustes Analysis**: Automatic initial direction calibration using orthogonal_procrustes
- **Coordinate Frame Alignment**: Transforms from local VIO frame to global coordinate system

**Advanced Features:**
- **Automatic Direction Initialization**: Uses first 4000 data points for robust direction estimation
- **Y-X Inversion Detection**: Automatically detects and corrects for inverted coordinate systems
- **Quaternion Handling**: Proper conversion from quaternion to yaw angles using Utils.quaternion_to_yaw
- **Seamless Mode Switching**: Maintains yaw continuity when switching from other estimation modes

**Implementation Details:**
- Merges VISO and GPOS data streams using timestamp-based alignment
- Calculates relative positions from first VISO measurement
- Applies rotation based on estimated initial direction
- Handles missing data gracefully with fallback to last known position

### 3. Ultra-Wideband (UWB)

UWB provides absolute positioning through sophisticated sensor fusion:

**Dual Protocol Support:**
- **UWBP (Position-based)**: Uses direction vectors and distance measurements
- **UWBT (Time-based)**: Employs azimuth, elevation angles with distance
- **Integration with GPOS**: Combines UWB measurements with GPS positioning data

**Advanced Positioning Algorithm:**
- **Multi-tag Priority System**: Prioritizes tags ("3637RLJ", "3636DWF", "3583WAA")
- **Weighted Average Estimation**: Uses accuracy-weighted position averaging
- **Confidence-based Fusion**: Combines estimates based on measurement reliability
- **Coordinate Transformation**: Applies quaternion rotations for proper coordinate mapping

**Sophisticated Accuracy Calculation:**
- **Time Difference Accuracy**: Sigmoid function for temporal synchronization quality
- **Distance-based Reliability**: Confidence scoring based on measurement range
- **NLOS Compensation**: Automatic detection and mitigation of non-line-of-sight conditions
- **Multi-factor Confidence**: Combines time, distance, and LOS factors using sigmoid functions

**Implementation Details:**
- Uses last 100 UWBT measurements for temporal analysis
- Applies scipy.spatial.transform.Rotation for proper 3D transformations
- Handles missing or invalid data with graceful degradation
- Maintains individual trajectory tracking for each UWB tag

## Implementation Details

### Estimation Modes

The system supports four distinct estimation modes controlled by `ESTIMATE_MODE` parameter:

1. **COMPETITION Mode** (Default)
   - Intelligent multi-sensor fusion with dynamic switching
   - Priority hierarchy: UWB (≥0.2 confidence) → VIO (>0.8 confidence) → PDR (fallback)
   - Automatic mode switching based on real-time confidence assessment
   - Maintains estimation continuity during mode transitions
   - Uses last known yaw angle for PDR initialization

2. **ONLY_PDR Mode** (Demo Only)
   - Pure inertial navigation using accelerometer and gyroscope
   - Automatic device orientation detection and compensation
   - Uses configurable initial angle (`init_angle_rad()`) for direction reference
   - Suitable for environments without visual or UWB infrastructure

3. **ONLY_VIO Mode** (Demo Only)
   - Vision-based tracking with GPS integration
   - Automatic initial direction calibration using Procrustes analysis
   - Graceful fallback to last known position when VIO data unavailable
   - Requires adequate visual features and GPS reference

4. **ONLY_UWB Mode** (Demo Only)
   - Pure UWB-based absolute positioning
   - Uses confidence threshold of ≥0.3 for position updates
   - Falls back to zero position when confidence insufficient
   - No drift accumulation, provides absolute position reference

### Sensor Fusion Strategy

The Competition mode fusion algorithm employs an intelligent hierarchical approach:

1. **UWB Priority Assessment**:
   - Calculate UWB accuracy using multi-factor confidence model
   - If confidence ≥ 0.2, use UWB position directly
   - Set current_method to "UWB" for status tracking

2. **VIO Secondary Assessment**:
   - Switch to VIO if not already active (initializes with last known pose)
   - Calculate VIO confidence based on data availability and quality
   - If confidence > 0.8, use VIO position
   - Set current_method to "VIO" for status tracking

3. **PDR Fallback Strategy**:
   - Switch to PDR if other methods insufficient (uses last yaw as initial direction)
   - Automatic device orientation detection and angle compensation
   - Always provides position estimate as ultimate fallback
   - Set current_method to "PDR" for status tracking

4. **Graceful Degradation**:
   - Uses VIO position if available when PDR confidence low
   - Maintains last known position as final fallback
   - Ensures continuous position output under all conditions

### Data Synchronization

Critical for multi-sensor fusion:
- Timestamp alignment across all sensors
- Interpolation for asynchronous data streams
- Buffer management for real-time processing
- Latency compensation mechanisms

## Performance Metrics

### Evaluation Framework

The system includes comprehensive evaluation capabilities:

1. **RMSE Calculation**
   - Compares estimates against ground truth
   - Per-axis and total error metrics
   - Temporal error analysis

2. **Visualization Tools**
   - Real-time trajectory plotting
   - Map overlay capabilities
   - Error heat maps
   - Sensor status monitoring

3. **Logging System**
   - Detailed debug information
   - Performance profiling
   - Error tracking and recovery logs

### Optimization Strategies

1. **Computational Efficiency**
   - Vectorized operations using NumPy
   - Efficient data structures (deque for sliding windows)
   - Lazy evaluation where possible
   - Multi-threading for parallel processing

2. **Memory Management**
   - Bounded buffers for sensor data
   - Automatic garbage collection
   - Efficient DataFrame operations with Pandas

3. **Real-time Constraints**
   - Configurable maximum wait times
   - Immediate mode for testing
   - Asynchronous processing pipelines

### Configuration and Tuning

The system uses a centralized parameter management system (`src/lib/params/`):

**Core Configuration Parameters:**
- `ESTIMATE_MODE`: Controls fusion strategy (COMPETITION/ONLY_PDR/ONLY_VIO/ONLY_UWB)
- `window_acc_sec()`, `window_gyro_sec()`: Smoothing windows for sensor data
- `peak_distance_sec()`, `peak_height()`: PDR step detection parameters
- `stride_scale()`, `stride_threshold()`: Step length estimation parameters
- `init_angle_rad()`: Initial direction for PDR mode
- `uwb_time_diff_k()`, `uwb_distance_k()`: UWB confidence model parameters

**Environment-based Configuration:**
- Demo vs. Competition mode parameter isolation
- Runtime parameter adjustment via `Params.set_param()`
- Mode-specific optimizations and thresholds
- Performance/accuracy trade-offs based on use case

### Error Handling and Recovery

Robust error handling ensures system reliability:

1. **API Communication Errors**:
   - Comprehensive HTTP status code handling (200, 404, 405, 422, 423)
   - Graceful handling of reload restrictions and timing constraints
   - Demo vs. competition mode error handling differences

2. **Sensor Data Failures**:
   - Missing data graceful degradation (returns zero confidence/fallback positions)
   - Invalid sensor type detection and logging
   - Timestamp synchronization error handling

3. **Computational Errors**:
   - Exception catching with state preservation in estimation methods
   - Interactive error recovery in demo mode
   - Silent error handling in competition mode to prevent interruption

4. **Mode Switching Safety**:
   - Proper initialization when switching between estimation modes
   - State preservation during mode transitions
   - Confidence-based failsafe mechanisms

## Conclusion

This indoor positioning system represents a comprehensive solution for the xDR Challenge 2025, combining state-of-the-art positioning technologies with robust software engineering practices. The modular architecture enables easy extension and adaptation, while the multi-sensor fusion approach ensures reliable positioning across diverse environments.

The system's strength lies in its ability to intelligently combine complementary positioning technologies, leveraging the strengths of each while mitigating their individual weaknesses. This results in a positioning solution that is both accurate and robust, suitable for real-world deployment in challenging indoor environments.