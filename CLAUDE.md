# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

# Guidelines

This document defines the project's rules, objectives, and progress management methods. Please proceed with the project according to the following content.

## Top-Level Rules

- To maximize efficiency, **if you need to execute multiple independent processes, invoke those tools concurrently, not sequentially**.
- **You must think exclusively in English**. However, you are required to **respond in Japanese**.
- To understand how to use a library, **always use the Contex7 MCP** to retrieve the latest information.

## Programming Rules

- Avoid hard-coding values unless absolutely necessary.
- Do not use `any` or `unknown` types in TypeScript.
- You must not use a TypeScript `class` unless it is absolutely necessary (e.g., extending the `Error` class for custom error handling that requires `instanceof` checks).


## Overview

This is an Indoor Positioning System (IPS) for the xDR Challenge 2025 competition. The system processes sensor data (accelerometer, gyroscope, magnetometer, UWB, GPS, visual-inertial odometry) and implements sensor fusion algorithms for real-time position estimation.

## Development Commands

### Setup
```bash
# Create virtual environment
python -m venv .venv

# Activate virtual environment
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Running the Application

```bash
# Demo mode (local EvAAL API server)
python main.py --demo

# Demo mode with local server
python main.py --demo --run-server

# Production mode
python main.py

# Immediate execution mode (demo only, no EvAAL API)
python main.py --demo --immediate
```

### Type Checking
```bash
# Run mypy for type checking
mypy src/
```

## Architecture

### Core Components

1. **Requester** (`src/lib/requester/`): Handles EvAAL API communication
   - Fetches sensor data from API endpoints
   - Manages data request intervals

2. **Recorder** (`src/lib/recorder/`): Processes different sensor types
   - ACCE, GYRO, MAGN: IMU sensors
   - AHRS: Attitude and Heading Reference System
   - UWBP/UWBT: Ultra-Wideband positioning/timing
   - GPOS: GPS data
   - VISO: Visual-Inertial Odometry

3. **Localizer** (`src/lib/localizer/`): Sensor fusion algorithms
   - PDR (Pedestrian Dead Reckoning)
   - UWB positioning
   - VIO (Visual-Inertial Odometry) - currently disabled

4. **Visualizer** (`src/lib/visualizer/`): Result visualization
   - Generates floor plan overlays
   - Creates estimation plots

### Data Flow

1. Main pipeline (`src/pipeline.py`) orchestrates the entire process
2. Requesters fetch sensor data from EvAAL API
3. Recorders process and store sensor measurements
4. Localizers perform sensor fusion to estimate position
5. Results are sent back to EvAAL API and visualized

### Key Files

- `src/type.py`: Type definitions and data structures
- `src/api/evaalapi.yaml`: EvAAL API configuration
- `.env.demo` / `.env.competition`: Environment configurations

## Important Notes

- The project uses strict type checking with mypy (`disallow_untyped_defs = True`)
- VIO localizer is temporarily disabled (see `src/lib/localizer/_vio.py`)
- Map files are stored in the `map/` directory
- Output (CSV files and plots) are saved to `output/` directory by default