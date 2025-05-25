<h1>Autonomous AI-Powered Cleanup RobotMimoAICleaningRobot</h1>
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
Targeted Object Detection:</strong><br>
When cleanup is required, the system identifies and locates specific objects (e.g., utensils, food items) using advanced object detection algorithms.<br>
<br>
<strong>Autonomous Manipulation:</strong><br>
The robot calculates precise distances and coordinates, enabling it to pick up and move objects efficiently and safely.<br>
<br>
Seamless Integration:</strong><br>
All components communicate over a robust TCP/IP protocol, ensuring reliable real-time control and feedback.<br>
<br>
<h2>Business Value</h2></strong><br>
<strong>Enhanced Productivity:</strong><br>
Automates routine cleanup tasks, allowing human staff to focus on higher-value activities.<br>
<br>
Improved Safety and Hygiene:<br>
Ensures workspaces remain clean and organized, reducing risks and promoting a healthier environment.<br>
<br>
Cutting-Edge Innovation:</strong><br>
Demonstrates leadership in AI, robotics, and human-machine collaboration, supporting digital transformation initiatives.<br>
<br>
This project exemplifies the integration of AI, robotics, and user-centric design to deliver tangible operational benefits and a showcase for next-generation automation.
This code is for a elephant robotics ai robot with 6 degrees of freedom that utilizes speech recognition combined with a multimodal transformer-based large language model. called bakllava. The robot calculates the distance to a target object that shall be cleaned up in case a human person is detected that says is it messy to trigger the cleanup. to detect if it is messy the baklava model is used. to locate the object to be cleaned up yolo v8 is used. the robot is connected via a proprietry tcp/ip socket library. Do write a sophisticated feature list in bullet points<br>
<br>
<strong>Feature List: Elephant Robotics AI Robot (6-DOF) with Multimodal AI and Speech Control</strong><br>
<br>
<strong>6 Degrees of Freedom Robotic Arm</strong><br>
Enables complex manipulation and precise movement in 3D space.<br>
<br>
<strong>Speech Recognition Integration</strong><br>
Uses speech_recognition and pyttsx3 for natural language interaction and voice feedback.<br>
<br>
<strong>Multimodal Transformer-based LLM (Bakllava)</strong><br>
<br>
Accepts both image and text input for context-aware reasoning.<br>
Determines scene cleanliness or messiness from camera images using advanced vision-language understanding.<br>
YOLOv8 Object Detection<br>
<br>
Detects and localizes specific objects (e.g., fork, banana) in real-time.<br>
Calculates object position and distance for robotic manipulation.<br>
Face Mesh Detection (MediaPipe)</strong><br>
<br>
Detects human presence and facial landmarks.<br>
Calculates distance to human face for safety and interaction context.<br>
Dynamic Task Triggering<br>
<br>
Initiates cleanup sequence when a human is detected and the phrase &quot;is it messy&quot; is spoken.<br>
Uses Bakllava model to analyze the scene and decide if cleanup is needed.<br>
Automated Cleanup Workflow</strong><br>
<br>
If mess is detected, robot locates target object using YOLOv8.<br>
Calculates 3D coordinates and distance to object for precise pickup.<br>
Controls gripper state (open/close) and arm movement for object manipulation.<br>
Proprietary TCP/IP Socket Communication<br>
<br>
Robust, low-level communication with the robot hardware for real-time control.<br>
Supports sending/receiving joint angles, coordinates, and gripper commands.<br>
Camera Calibration and Image Preprocessing<br>
<br>
Supports multiple resolutions and cropping for optimal AI inference.<br>
Converts coordinates between camera, YOLO, and robot coordinate systems.<br>
Error Handling and State Management<br>
<br>
Handles connection errors, movement limits, and synchronization issues.<br>
Maintains state flags for asynchronous actions and task sequencing.<br>
Extensible Modular Design<br>
<br>
Clear separation of detection, inference, and robot control logic.<br>
Easily adaptable for new tasks, models, or hardware.<br>
Real-time Visual Feedback<br>
<br>
Displays camera feed with overlays for detected faces, objects, and landmarks.<br>
Annotates images with detection results and distances.<br>
Logging and Response Tracking</strong><br>
<br>
Logs AI model responses and robot actions for debugging and analysis.<br>
This system enables a highly interactive, intelligent, and autonomous robot capable of understanding and acting on complex multimodal commands in real-world environments.<br>
<br>
<br>
<br>
<br>
<h2>Initialization and Setup</h2><br>
<br>
Imports a wide range of libraries for image processing (cv2, PIL), speech recognition (speech_recognition, pyttsx3), networking (socket), AI inference (ultralytics.YOLO, Bakllava LLM), and custom hardware control.<br>
Sets up camera and image resolution constants for different AI models (Bakllava, YOLO, ROS).<br>
Initializes the robot&rsquo;s TCP/IP socket connection via a custom MycobotServer class for sending/receiving joint angles, coordinates, and gripper commands.<br>
<h1>Vision Modules</h1><br>
<br>
<strong></stronn></strongn>YOLOv8 Object Detection:</strong><br>
The YoloInferencer class loads a YOLOv8 model and provides a predict method to detect objects (e.g., fork, banana) in camera frames, returning bounding boxes and confidence scores.<br>
<strong>Face Mesh Detection:</strong><br>
The FaceMeshDetector class uses MediaPipe to detect facial landmarks, enabling calculation of the distance to a human face using pixel geometry and known anthropometric distances (e.g., interpupillary distance).<br>
Speech and Multimodal AI<br>
<br>
Uses speech_recognition to listen for human speech. When a person is detected and the phrase &quot;is it messy&quot; is spoken, the system triggers a scene analysis.<br>
The scene image is processed and sent (as base64) to a Bakllava multimodal LLM via HTTP POST. The model&rsquo;s response determines if the environment is messy.<br>
Cleanup Trigger and Object Localization<br>
<br>
If Bakllava&rsquo;s response indicates the scene is messy, the system activates YOLOv8 to locate target objects for cleanup.<br>
Calculates the 3D position and distance to the object using bounding box size and camera calibration, then translates these to robot coordinates.<br>
Robot Control and Manipulation<br>
<br>
The robot arm is controlled via TCP/IP commands to move to the target object, open/close the gripper, and perform cleanup actions.<br>
Implements feedback loops for movement correction, using visual feedback to iteratively adjust the robot&rsquo;s position relative to the object.<br>
Includes safety and state management, such as checking if the gripper is moving, handling connection errors, and synchronizing actions.<br>
User Feedback and Logging<br>
<br>
Provides real-time visual feedback via OpenCV windows, overlaying detection results and distances.<br>
Uses text-to-speech to inform the user of actions and AI decisions.<br>
Logs AI responses and robot actions for traceability.<br>
Technical Highlights<br>
<strong>Multimodal Reasoning:</strong><br>
Combines visual (YOLO, FaceMesh) and language (Bakllava LLM) AI for context-aware decision-making.<br>
<strong>Real-Time Control:</strong><br>
Uses a proprietary TCP/IP protocol for low-latency robot actuation.<br>
<strong>Adaptive Feedback:</strong><br>
Continuously refines robot movement based on visual feedback and error correction.<br>
<strong>Extensible Design:</strong><br>
Modular classes for detection, inference, and robot control allow for easy upgrades or task changes.<br>
<strong>Summary:</strong><br>
This code enables a 6-DOF robot to autonomously detect human presence, understand spoken commands, assess environmental cleanliness using a multimodal LLM, locate objects with YOLOv8, and perform cleanup actions&mdash;all with real-time feedback and robust error handling.<br>
<br>
