<h1>MimoAICleaningRobot</h1>
An AI-powered Elephant Robotics myCobot 280 NVIDIA Jetson Nano 6-DOF robotic system that uses speech recognition, multimodal vision-language models, and real-time object detection to autonomously detect, assess, and clean up messy environments through intelligent human-robot interaction.
<br>
<h2>Autonomous AI-Powered Cleanup Robot</h2>
<strong>Project Overview:</strong><br>
This project delivers an advanced, AI-driven robotic solution for workspace cleanliness and human-robot interaction. The system leverages a 6-axis Elephant Robotics arm, state-of-the-art computer vision, and natural language processing to autonomously detect, assess, and respond to environmental messiness.<br>
<br>
<h2>Key Capabilities</h2><br>
<strong>Human Interaction:</strong><br>
The robot detects human presence and responds to spoken commands, enabling intuitive, hands-free operation.<br>
<br>
<strong>Intelligent Scene Understanding:</strong><br>
Using a multimodal transformer-based AI model, the robot analyzes visual input and determines if the environment is messy, ensuring context-aware decision-making.<br>
<br>
<strong>Targeted Object Detection:</strong><br>
When cleanup is required, the system identifies and locates specific objects (e.g., utensils, food items) using advanced object detection algorithms.<br>
<br>
<strong>Autonomous Manipulation:</strong><br>
The robot calculates precise distances and coordinates, enabling it to pick up and move objects efficiently and safely.<br>
<br>
<strong>Seamless Integration:</strong><br>
All components communicate over a robust TCP/IP protocol, ensuring reliable real-time control and feedback of the robots ROS2 MoveIt Inverse Kinematic Solver TRAC_IKKinematicsPlugin.<br>
<br>
<h2>Business Value</h2></strong><br>
<strong>Enhanced Productivity:</strong><br>
Automates routine cleanup tasks, allowing human staff to focus on higher-value activities.<br>
<br>
<strong>Improved Safety and Hygiene:</strong><br>
Ensures workspaces remain clean and organized, reducing risks and promoting a healthier environment.<br>
<br>
<strong>Cutting-Edge Innovation:</strong><br>
This project exemplifies the integration of AI, robotics, and user-centric design to deliver tangible operational benefits and a showcase for next-generation automation.
This code is for a elephant robotics AI robot with 6 degrees of freedom that utilizes speech recognition combined with a multimodal transformer-based large language model, called bakllava.
<br>
<h2>Feature List: Elephant Robotics AI Robot (6-DOF) with Multimodal AI and Speech Control</h2><br>
<strong>6 Degrees of Freedom Robotic Arm</strong><br>
Enables complex manipulation and precise movement in 3D space.<br>
Utilizes the ROS2 MoveIt motion planning framework with collision detection and and avoidance visualized in RViz.<br>
<br>
<strong>Speech Recognition Integration</strong><br>
Uses speech_recognition and pyttsx3 for natural language interaction and voice feedback.<br>
<br>
<strong>Multimodal Transformer-based LLM (Bakllava)</strong><br>
Accepts both image input for context-aware reasoning.<br>
Determines scene cleanliness or messiness from camera images using advanced vision-language understanding.<br>
<br>
<strong>YOLO (Object Detection)</strong><br>
<br>
Detects and localizes specific objects (e.g., fork, banana) in real-time.<br>
Calculates object position and distance for robotic manipulation.<br>
<br>
<strong>Face Mesh Detection (MediaPipe)</strong><br>
Detects human presence and facial landmarks.<br>
Calculates distance to human face for safety and interaction context.<br>
<br>
<strong>Dynamic Task Triggering</strong><br>
Initiates cleanup sequence when a human is detected and the phrase &quot;is it messy&quot; is spoken.<br>
Uses Bakllava model to analyze the scene and decide if cleanup is needed.<br>
<br>
<strong>Automated Cleanup Workflow</strong><br>
If mess is detected, robot locates target object using YOLO.<br>
Calculates 3D coordinates and distance to object for precise pickup.<br>
Controls gripper state (open/close) and arm movement for object manipulation.<br>
Utilizes proprietary TCP/IP Socket Communication: <a href="https://github.com/BierschneiderEmanuel/MyCobot280BSDSocketInterface.git">MyCobot280 BSD Socket Interface Repository</a><br>
<br>
Robust, low-level communication with the robot hardware for real-time control.<br>
Supports sending/receiving joint angles, coordinates, and gripper commands.<br>
<br>
<strong>Camera Calibration and Image Preprocessing</strong><br>
Supports multiple resolutions and cropping for optimal AI inference.<br>
Converts coordinates between camera, YOLO, and robot coordinate systems.<br>
<br>
<strong>Error Handling and State Management</strong><br>
Handles connection errors, movement limits, and synchronization issues.<br>
Maintains state flags for asynchronous actions and task sequencing.<br>
<br>
<strong>Extensible Modular Design</strong><br>
Clear separation of detection, inference, and robot control logic.<br>
Easily adaptable for new tasks, models, or hardware.<br>
<br>
<strong>Real-time Visual Feedback</strong><br>
Displays camera feed with overlays for detected faces, objects, and landmarks.<br>
Annotates images with detection results and distances.<br>
<br>
<strong>Logging and Response Tracking</strong><br>
Logs AI model responses and robot actions for debugging and analysis.<br>
This system enables a highly interactive, intelligent, and autonomous robot capable of understanding and acting on complex multimodal commands in real-world environments.<br>
<br>
<h1>Initialization and Setup</h1><br>
Imports a wide range of libraries for image processing (cv2, PIL), speech recognition (speech_recognition, pyttsx3), networking (socket), AI inference (ultralytics YOLO, Bakllava LLM), and custom hardware control.<br>
Sets up camera and image resolution constants for different AI models (Bakllava, YOLO, ROS).<br>
Initializes the robot&rsquo;s TCP/IP socket connection via a custom MycobotServer class for sending/receiving joint angles, coordinates, and gripper commands.<br>
<br>
<strong>YOLO Object Detection</strong><br>
The YoloInferencer class loads a YOLO model and provides a predict method to detect objects (e.g., fork, banana) in camera frames, returning bounding boxes and confidence scores.<br>
<br>
<strong>Face Mesh Detection</strong><br>
The FaceMeshDetector class uses MediaPipe to detect facial landmarks, enabling calculation of the distance to a human face using pixel geometry and known anthropometric distances (e.g., interpupillary distance).<br>
<br>
<strong>Speech and Multimodal AI</strong><br>
Uses speech_recognition to listen for human speech. When a person is detected and the phrase &quot;is it messy&quot; is spoken, the system triggers a scene analysis.<br>
The scene image is processed and sent (as base64) to a Bakllava multimodal LLM via HTTP POST. The model&rsquo;s response determines if the environment is messy.<br>
<br>
<strong>Cleanup Trigger and Object Localization</strong><br>
If Bakllava&rsquo;s response indicates the scene is messy, the system activates YOLO to locate target objects for cleanup.<br>
Calculates the 3D position and distance to the object using bounding box size and camera calibration, then translates these to robot coordinates.<br>
<br>
<strong>Robot Control and Manipulation</strong><br>
The robot arm is controlled via TCP/IP commands to move to the target object, open/close the gripper, and perform cleanup actions.<br>
Implements feedback loops for movement correction, using visual feedback to iteratively adjust the robot&rsquo;s position relative to the object.<br>
Includes safety and state management, such as checking if the gripper is moving, handling connection errors, and synchronizing actions.<br>
<br>
<strong>User Feedback and Logging</strong><br>
Provides real-time visual feedback via OpenCV windows, overlaying detection results and distances.<br>
Uses text-to-speech to inform the user of actions and AI decisions.<br>
Logs AI responses and robot actions for traceability.<br>
<br>
<h2>Technical Highlights</h2>
<strong>Multimodal Reasoning:</strong><br>
Combines visual (YOLO, FaceMesh) and language (Bakllava LLM) AI for context-aware decision-making.<br>
<strong>Real-Time Control:</strong><br>
Uses a proprietary TCP/IP protocol for low-latency robot actuation.<br>
<strong>Adaptive Feedback:</strong><br>
Continuously refines robot movement based on visual feedback and error correction.<br>
<strong>Extensible Design:</strong><br>
Modular classes for detection, inference, and robot control allow for easy upgrades or task changes.<br>
<strong>Summary:</strong><br>
This code enables a 6-DOF robot to autonomously detect human presence, understand spoken commands, assess environmental cleanliness using a multimodal LLM, locate objects with YOLO, and perform cleanup actions&mdash;all with real-time feedback and robust error handling.<br>
<br>
