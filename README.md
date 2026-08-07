Reality Painter

Draw in the air using hand tracking powered by Reality Engine.

Features
Drawing
Air drawing
Smooth stroke interpolation
Cursor trail
Brush preview
Brushes
Hard Brush
Soft Brush
Marker
Highlighter
Shapes
Line
Rectangle
Circle
Editing
Undo
Redo
Clear canvas
Eraser
Saving
Save artwork as an image
Interface
Radial menu
Developer HUD
User HUD
Live cursor visualization
Getting Started
Requirements
Webcam
Python
Required dependencies from requirements.txt
Run
python apps/reality_painter/app.py
Controls
Hand Controls
Action	Description
Move index finger	Move cursor
Draw gesture (pinch/click, depending on your current configuration)	Draw
Release	Stop drawing

(Describe the actual draw gesture exactly as your implementation uses it.)

Keyboard Shortcuts
Key	Action
Q	Open / Close Radial Menu
B	Cycle Brush
G	Cycle Shape
E	Toggle Eraser
U	Undo
R	Redo
C	Clear Canvas
S	Save Canvas
[	Decrease Brush Size
]	Increase Brush Size
1–5	Change Color
H	Toggle Developer HUD (if implemented)
ESC	Exit
Using the Radial Menu
Press Q.
The menu opens at the current cursor position.
Move your hand over a menu item.
Pinch (or perform the configured selection gesture) to confirm.
The menu closes automatically.
Continue drawing.
Project Architecture
Reality Engine
│
├── Vision
├── Tracking
├── Interaction
├── Rendering
└── Pipeline

↓

Reality Painter

├── sketch.py
├── brushes.py
├── shapes.py
├── menu.py
└── app.py
Current Release

Version: v2.0.0

Status: Stable

Codename: Reality Painter Phase 10