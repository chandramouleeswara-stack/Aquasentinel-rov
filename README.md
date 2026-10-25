# AquaSentinel — Underwater ROV for Inspection, Marine Debris Detection & Water Monitoring

<h2>Team Details</h2>
<b>Team Number: </b><p>26AACR14</p>
<b>Senior Mentor:</b><p>Aashrith</p>
<b>Junior Mentors:</b><p>Dharvik, Nikhitha</p>
<b>Team Member 1:</b><p>Dhanalaxmi Kulkarni</p>
<b>Team Member 2:</b><p>Shubhanu Ojha</p>
<b>Team Member 3:</b><p>Nikhil</p>
<b>Team Member 4:</b><p>Pavan</p>
<b>Team Member 5:</b><p>Chandramouleeswara Sarma</p>

<div align="center">
<h2> Abstract </h2>
</div>

AquaSentinel is a low-cost IoT underwater ROV for shallow-water inspection, real-time marine debris detection and environmental monitoring. The vehicle is tethered and remotely operated. It uses four propeller-driven thrusters, each built from a 1100 GPH bilge pump motor, controlled through BTS7960 motor drivers by an Arduino Mega 2560. An onboard sensor suite — an IMU, a temperature probe, a turbidity sensor and a DYP-L08 underwater ultrasonic distance sensor — collects data on its surroundings. A waterproof camera streams live video to the surface over a tether, and the operator controls the ROV from a laptop using a gamepad/joystick.

The project is built in three layers:
- **IoT:** camera, turbidity, temperature, distance and motion data streamed to a dashboard.
- **ML:** debris/anomaly detection using camera data and environmental context.
- **MLOps:** monitoring, drift detection, model versioning and retraining.

The MLOps layer turns a one-off detector into a system that keeps working in new water bodies, seasons, and lighting/turbidity conditions. It is a feedback loop: the ROV collects data, the model predicts, metrics and drift are tracked, retraining is triggered, and the new model is validated and deployed back to the ROV/dashboard. The project is a low-cost prototype aimed at demonstrating core ROV functions (propulsion, stabilization, live video, underwater sensing) plus a maintainable AI pipeline, within a hardware build budget of roughly ₹13,000–14,000.

## Table of Contents
- [Introduction](#introduction)
- [Requirements](#requirements)
- [Installation and Usage](#installation-and-usage)
- [System Architecture](#system-architecture)
- [Repository Structure](#repository-structure)
- [MLOps Layer](#mlops-layer)
  - [Part 1: How the MLOps Layer Works](#part-1-how-the-mlops-layer-works)
    - [Core Components](#core-components)
    - [Fit with the Existing Architecture](#fit-with-the-existing-architecture)
    - [Minimum Viable MLOps (Start Here)](#minimum-viable-mlops-start-here)
  - [Part 2: Finding Datasets](#part-2-finding-datasets)
    - [Public Research Datasets](#public-research-datasets)
    - [Ready-to-Use Platforms](#ready-to-use-platforms)
    - [Collecting Your Own Data](#collecting-your-own-data-critical-for-domain-fit)
    - [Recommended Dataset Strategy](#recommended-dataset-strategy)
- [Project Plan](#project-plan)
- [Improvements](#improvements)
- [Contribution](#contribution)

## Introduction

Inspecting underwater structures, water bodies, or confined aquatic spaces is often slow, risky, or requires divers. AquaSentinel is a small tethered ROV that lets an operator explore underwater areas from the surface, see a live video feed, detect marine debris, and log basic water-quality and motion data as it moves. It's designed as a student project to learn ROV fundamentals — buoyancy, thrust, waterproofing, sensor integration, and surface-to-vehicle control — as well as how to deploy and maintain a machine-learning model in the field, on a limited budget.

**Key Features:**
- **4x Propeller Thrusters:** 1100 GPH bilge pump motors fitted with propellers, driven by BTS7960 drivers for forward/reverse and turning control underwater
- **MPU6050 IMU:** Tracks orientation and tilt to help monitor and stabilize the ROV's attitude
- **DS18B20 Waterproof Probe:** Measures water temperature
- **Turbidity Sensor:** Measures water clarity
- **DYP-L08 Underwater Ultrasonic Sensor:** Detects obstacles/seabed distance underwater
- **Waterproof USB Camera:** Streams live video to the surface laptop
- **Gamepad/Joystick Control:** Operator drives the ROV from the surface via USB input
- **Tethered Operation:** Control commands, sensor data and video travel through a surface tether, giving real-time control with no signal lag from wireless links
- **3S LiPo Powered:** A 3S LiPo battery supplies the high current the thrusters need in a compact package
- **YOLO Marine Debris Detection:** Real-time debris/anomaly detection from camera frames, using sensor readings (turbidity, temperature, depth, distance) as environmental context
- **MLOps Feedback Loop:** Data pipeline and labeling, model registry and versioning, inference monitoring, drift detection, a retraining pipeline with evaluation gates, canary deployment with rollback, and operator feedback

## Requirements

### Hardware
- **Arduino Mega 2560** (main controller)
- **4x 1100 GPH bilge pump motors** (thruster motors)
- **4x Propellers** (one per thruster motor)
- **4x BTS7960 motor driver modules** (high-current H-bridge)
- **3S LiPo battery** (11.1V nominal, 12.6V fully charged)
- **Main power switch**
- **MPU6050** 6-axis accelerometer/gyroscope
- **DS18B20** waterproof temperature sensor + 4.7kΩ pull-up resistor
- **Turbidity sensor**
- **DYP-L08** underwater ultrasonic distance sensor
- **Waterproof USB endoscope camera**
- **Sealable waterproof enclosure, cable glands, silicone sealant**
- **PVC frame, floats, and ballast weights**
- **Multi-core tether cable**
- **USB gamepad/joystick**
- **Laptop (surface station)**
- **Connecting wires**
- **Optional edge gateway** (Jetson / Raspberry Pi) for running the model close to the ROV

### Software
- **Arduino IDE** (version 1.8.x or higher)
- **Python 3.7+** (for the surface control script and the ML/MLOps scripts)
- Python libraries for camera display and gamepad input on the surface laptop
- **YOLO training code** (YOLOv5 / v8 / v11) and trained weights for the debris detector
- **MLOps tooling** (see [Core Components](#core-components) for the full list):
  - Experiment tracking and model registry: MLflow or Weights & Biases
  - Labeling: CVAT, Roboflow or LabelImg
  - Retraining automation: a scheduled script, GitHub Actions or GitLab CI
  - Database + dashboard for prediction logs and monitoring

### Arduino Libraries
- Wire (I2C communication)
- Adafruit_MPU6050 (also install its dependencies: Adafruit_Sensor and Adafruit_BusIO)
- OneWire + DallasTemperature (DS18B20)

## Installation and Usage

This section covers the ROV hardware and firmware. To set up the AI detection, monitoring and retraining layer, see [MLOps Layer](#mlops-layer).

### Step 1: Clone the Repository
```bash
git clone https://github.com/<your-username>/AquaSentinel-ROV.git
cd AquaSentinel-ROV
```

### Step 2: Hardware Setup
1. Wire the components according to the circuit diagram in `/circuit`:
   - BTS7960 Motor Drivers (x4): RPWM/LPWM to Arduino PWM pins, R_EN/L_EN enabled (to digital pins or 5V), VCC to 5V, B+/B− to the 3S LiPo supply (through the main power switch), M+/M− to each thruster motor
   - Propellers: fit one propeller securely on each motor shaft
   - MPU6050: SDA→D20, SCL→D21
   - DS18B20: Data→(assigned pin) with 4.7kΩ pull-up to 5V
   - Turbidity sensor: Analog out→A0
   - DYP-L08 ultrasonic sensor: connect to the Arduino per its output mode (e.g., TX/RX→Serial pins)
2. Ensure all grounds are common across the Arduino, drivers, and sensors.
3. Seal all wiring joints and enclosure penetrations before any water testing.

### Step 3: Upload Code to the Arduino
1. Open the sketch in `/firmware` in Arduino IDE
2. Select Board: "Arduino Mega 2560"
3. Select the correct COM port
4. Upload the sketch

### Step 4: Surface Station Setup
1. Connect the tether/USB link between the ROV and the laptop
2. Plug in the gamepad/joystick
3. Run the surface control script to read gamepad input and send motor commands
4. Open the camera feed from the endoscope (via its USB connection)

### Step 5: Pre-Dive Checks
- Bucket-test the sealed enclosure for leaks before every dive
- Check the LiPo is fully charged and not swollen or damaged; avoid draining it below about 3.3V per cell (9.9V for the pack)
- Confirm propellers are tight and spin freely in the correct direction
- Confirm thrusters respond correctly to control input
- Confirm sensor readings are updating on the serial monitor / dashboard
- Confirm the detection model version in use is recorded for the mission
- Keep dives within the tested depth and current-draw limits of the build

## System Architecture

```
┌─────────────────────────────────────────┐
│              AQUASENTINEL ROV            │
│  ┌───────────────┐   ┌────────────────┐ │
│  │ Arduino Mega   │   │ Sensors:        │ │
│  │ 2560           │←→│ MPU6050         │ │
│  │                │   │ DS18B20         │ │
│  │ - 4x BTS7960   │   │ Turbidity       │ │
│  │   Drivers      │   │ DYP-L08         │ │
│  │                │   │ Ultrasonic      │ │
│  └───────┬────────┘   └────────────────┘ │
│          │                                │
│  ┌───────┴────────┐                       │
│  │ 4x Bilge Pump   │                      │
│  │ Motors +        │                      │
│  │ Propellers      │                      │
│  └────────────────┘                       │
│  ┌────────────────┐                       │
│  │ USB Endoscope   │                      │
│  │ Camera          │                      │
│  └────────────────┘                       │
│  ┌────────────────┐                       │
│  │ 3S LiPo Battery │                      │
│  └────────────────┘                       │
└──────────────────┬────────────────────────┘
                    │ Tether (data/USB)
                    ↓
        ┌───────────────────────┐
        │     Surface Station    │
        │        (Laptop)        │
        │                         │
        │ - Gamepad/joystick      │
        │   input                 │
        │ - Live camera feed      │
        │ - Sensor data display   │
        │ - Sends frames + sensor │
        │   metadata to MLOps     │
        └───────┬─────────▲───────┘
                │         │ new model (canary → fleet)
                ↓         │
        ┌──────────────────────────────────┐
        │          MLOps Layer              │
        │                                    │
        │ - Data store (raw + labeled)       │
        │ - Model registry + versioning      │
        │ - Inference monitoring dashboard   │
        │ - Drift detection                  │
        │ - Retraining pipeline              │
        │ - Deployment + rollback            │
        │ - Operator feedback UI             │
        └──────────────────────────────────┘
```

The MLOps layer adds a data store for raw and labeled frames with sensor metadata, a model registry, monitoring services that consume prediction logs, a retraining pipeline, and a deployment mechanism around the existing IoT and ML layers. It can run on the surface laptop or an edge gateway (Jetson / Raspberry Pi) for demos, or on a server for always-on operation. See [Fit with the Existing Architecture](#fit-with-the-existing-architecture).

## Repository Structure

This project is hosted on **GitHub** as a public repository. The repo contains:

```
AquaSentinel-ROV/
├── firmware/              # Arduino sketch(es) for the onboard Mega 2560
├── circuit/               # wiring diagrams / Cirkit Designer exports, pin tables
├── docs/                  # parts list, BOM, and build notes
├── Img/                   # build photos
├── data/
│   ├── raw/               # frames/clips + sensor metadata (time, position, ROV and mission ID)
│   ├── labeled/           # versioned "golden" datasets in YOLO format
│   └── DATASET_V1.md      # dataset version notes
├── models/                # weights per version (e.g. aquasentinel-debris:v1.3)
├── logs/                  # prediction logs (model version, class, confidence, box, sensor data)
├── train.py               # reproducible training script, tied to a Git commit
├── drift_check.py         # data-drift and concept-drift checks
├── retrain_pipeline.py    # retraining + evaluation gates + promotion
├── .github/workflows/     # optional CI for scheduled retraining
└── README.md              # this file
```

GitHub was chosen because it's free, lets the whole team and mentors track changes and contribute, and is the standard way to share a student hardware project for review. No paid hosting or server is needed for the code itself — GitHub simply stores and version-controls the firmware, diagrams, ML code, and documentation.

## MLOps Layer

### Part 1: How the MLOps Layer Works

MLOps turns a one-off detector into a system that keeps working in new water bodies, seasons, and lighting/turbidity conditions. It is a feedback loop:

**ROV collects data → model predicts → metrics and drift are tracked → retraining is triggered → new model is validated and deployed back to the ROV/dashboard.**

#### Core Components

| Component | What it does | Suggested tools |
|---|---|---|
| 1. Data pipeline and labeling | Store frames/clips with metadata (time, position, sensor readings, ROV and mission ID). Pre-label with the current model, send low-confidence frames to a human, and curate versioned "golden" datasets across clear/turbid, day/night, depths and locations. | Raw data lake, CVAT, Roboflow, LabelImg |
| 2. Model registry and versioning | Track weights, config, dataset version, hyperparameters, Git commit and metrics (mAP, precision/recall, latency, size). Every model gets a version such as `aquasentinel-debris:v1.3` so each detection is traceable. | MLflow, Weights and Biases |
| 3. Inference monitoring | Log per-prediction model version, class, confidence, box and input metadata (turbidity, temperature, depth, time). Aggregate confidence distribution, per-class detection rate, latency and edge resource use. | Database + dashboard |
| 4. Drift detection | **Data drift:** brightness, colour histograms, turbidity distribution vs the training baseline (KL divergence, Wasserstein distance). **Concept drift:** shifts in class proportions, more low-confidence detections, operator-flagged errors. Flag "retrain recommended" past thresholds. | Python scripts, statistical tests |
| 5. Retraining pipeline | Triggered by drift or enough new labels. Train from a reproducible script tied to a Git commit, then pass evaluation gates: test on diverse held-out data and compare against production on mAP, false-positive rate and edge inference time. Promote only if thresholds pass. | Scheduled script or GitHub Actions / GitLab CI |
| 6. Deployment and rollback | Push the model to the ROV or an edge gateway with hot-swapping. Use canary rollout (one ROV first, then the fleet). Keep earlier versions and revert if metrics drop. The dashboard shows the active model version per ROV. | Model server, edge runtime (Jetson / Raspberry Pi) |
| 7. Operator feedback | A simple interface to mark false positives/negatives and add new classes (e.g. oil slick, fishing line). Feeds short-term rules and long-term retraining with corrected labels. | Labeling UI |

#### Fit with the Existing Architecture

MLOps adds a data store for raw and labeled frames with sensor metadata, a model registry, monitoring services that consume prediction logs, a retraining pipeline, a deployment mechanism and a feedback UI around the existing IoT and ML layers.

#### Minimum Viable MLOps (Start Here)

- Log every detection with model version, confidence and sensor readings in a database.
- **Weekly:** sample frames from different conditions and label about 200–500 images.
- **Monthly, or when performance drops:** retrain and track versions in MLflow or even a spreadsheet.
- Define 2–3 drift signals (average confidence, turbidity distribution shift) with simple thresholds meaning "time to retrain".
- Keep a clear process for swapping models on the ROV and recording which version ran each mission.

### Part 2: Finding Datasets

Three routes: public research datasets, ready-to-use platform datasets, and your own collected data (most valuable for MLOps and domain adaptation to Indian water bodies).

#### Public Research Datasets

| Dataset | Size and classes | Notes and access |
|---|---|---|
| Seaclear Marine Debris Dataset | ~8,610 images, ~40 categories (litter, animals, plants, robot parts) | ROV-captured shallow water with instance segmentation and detection labels. Closest match to your use case. Find via the paper or project repository. |
| TrashCan | ~7,200–7,600+ images; plastic bags, bottles, nets, plus fauna/flora | Built for YOLO-style underwater garbage detection. Often mirrored on Roboflow. Derived from JAMSTEC deep-sea footage. |
| JAMSTEC Deep-sea Debris Database | Tens of thousands of images/videos | Deep-sea conditions differ from shallow ROV missions; good for general debris appearance. godac.jamstec.go.jp/dsdebris |
| Neural Ocean (Roboflow) | ~5,127 images, 15 classes (masks, cans, glass/plastic bottles, nets, polythene bags, tires, etc.) | Strong match for a 15-class garbage detector. Search "Neural Ocean" on Roboflow Universe. |
| Others | ICRA19 (~7,600), DeepPlastic (~3,200), small custom sets (1,000–5,000) | Described in papers; subsets sometimes on GitHub or institutional pages. |

#### Ready-to-Use Platforms

- **Roboflow Universe:** search "underwater garbage detection", "underwater object detection", "marine debris". Many sets (7k–12k images) come in YOLO format, already split into train/val/test. Ideal for quick prototyping.
- **Kaggle:** search "underwater trash dataset" or "marine debris dataset" for curated TrashCan/Roboflow subsets and student datasets; notebooks let you experiment before downloading.
- **GitHub:** search "underwater debris dataset", "underwater garbage YOLO", "TrashCan dataset" for datasets (often Google Drive links) and YOLOv5/v8/v11 training code you can adapt.

#### Collecting Your Own Data (Critical for Domain Fit)

- **Collect:** record short videos on test runs, sample frames at 1–5 fps, and vary turbidity, lighting, depth and common local debris (bags, bottles, packaging, fishing gear). Log turbidity, temperature and distance with every clip.
- **Label:** start with 500–2,000 frames using bounding boxes (or masks) with classes like `plastic_bag`, `plastic_bottle`, `fishing_net`, `can`, `glass_bottle`, `other`. Tools: LabelImg, CVAT, Roboflow Annotate, MakeSense. Export in YOLO format.
- **Use in MLOps:** treat your local data as the production distribution. Pretrain on public data, fine-tune on yours, and keep adding hard frames (low confidence, false positives, new debris) as the continuous retraining set.

#### Recommended Dataset Strategy

| Step | Action |
|---|---|
| 1. Start public | Pick TrashCan, Neural Ocean or a Roboflow/Kaggle underwater garbage set and train a baseline YOLO model. |
| 2. Add domain data | Collect 1,000–3,000 frames from local tests, label about 1,000, and fine-tune on the mixed dataset. |
| 3. Iterate with MLOps | Deploy, log predictions and sensor metadata, sample and label hard cases, retrain, and version datasets alongside models. |

**Next decision:** choose detection (bounding boxes) or segmentation (masks), and confirm your priority classes, before picking a dataset.

## Project Plan

- Track all firmware changes with commit history, so mentors can see progress over time
- Keep the circuit diagram and parts list version-controlled alongside the code
- Version datasets alongside models, and record which model version ran on each mission
- Document build steps, test results (bucket tests, thrust tests, sensor calibration), and known issues
- Use GitHub Issues to track problems and planned fixes
- Update this README as the design changes (e.g., switching components or finalizing the tether design)

## Improvements
- **Ballast:** Add adjustable ballast for neutral trim at different payloads
- **New classes:** Extend the detector with new classes (e.g. oil slick, fishing line) through operator feedback
- **Domain adaptation:** Keep fine-tuning on locally collected data from Indian water bodies

## Contribution
**This section provides instructions and details on how to submit a contribution via a pull request. It is important to follow these guidelines to make sure your pull request is accepted.**

1. Before proposing changes, read through this README to understand the project's goals and current design. Pull requests should align with the project's purpose.
2. Match the programming language and library versions already used in the project.
3. Document any proposed changes: what problem you noticed, what you're changing, why, and how you tested it.
