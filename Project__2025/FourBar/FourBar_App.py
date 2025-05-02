#region imports
from FourBar_GUI import Ui_Form
from FourBarLinkage_MVC import FourBarLinkage_Controller
import PyQt5.QtGui as qtg
import PyQt5.QtCore as qtc
import PyQt5.QtWidgets as qtw
import math
import sys
import numpy as np
import scipy as sp
from scipy import optimize
from copy import deepcopy as dc
import logging
#endregion

# Set up logging to capture errors and debugging information
logging.basicConfig(
    filename='fourbar_debug.log',
    level=logging.DEBUG,
    format='%(asctime)s:%(levelname)s:%(message)s'
)

#region class definitions
class MainWindow(Ui_Form, qtw.QWidget):
    """Main window class for the Four-Bar Linkage simulation application.

    This class sets up the UI, connects signals to slots, and handles user interactions
    such as mouse events, simulation control, and parameter adjustments.

    Attributes:
        FBL_C (FourBarLinkage_Controller): Controller for the four-bar linkage model and view.
        mouseDown (bool): Tracks whether the mouse is currently pressed.
    """

    def __init__(self):
        """Initialize the main window, set up the UI, and connect signals to slots."""
        super().__init__()
        self.setupUi(self)  # Set up the UI from the designer file

        #region UserInterface stuff here
        # Create and add the "Min Angle" control
        self.nud_MinAngle = qtw.QDoubleSpinBox(self)
        self.nud_MinAngle.setRange(-360, 360)  # Allow full rotation range
        self.nud_MinAngle.setValue(0)  # Default min angle
        self.nud_MinAngle.setObjectName("nud_MinAngle")
        self.horizontalLayout.addWidget(self.nud_MinAngle)
        self.lbl_MinAngle = qtw.QLabel("Min Angle", self)
        self.horizontalLayout.insertWidget(self.horizontalLayout.indexOf(self.nud_MinAngle), self.lbl_MinAngle)

        # Create and add the "Max Angle" control
        self.nud_MaxAngle = qtw.QDoubleSpinBox(self)
        self.nud_MaxAngle.setRange(-360, 360)
        self.nud_MaxAngle.setValue(180)  # Default max angle
        self.nud_MaxAngle.setObjectName("nud_MaxAngle")
        self.horizontalLayout.addWidget(self.nud_MaxAngle)
        self.lbl_MaxAngle = qtw.QLabel("Max Angle", self)
        self.horizontalLayout.insertWidget(self.horizontalLayout.indexOf(self.nud_MaxAngle), self.lbl_MaxAngle)

        # Create and add the "Damping Coefficient" control
        self.nud_Damping = qtw.QDoubleSpinBox(self)
        self.nud_Damping.setRange(0, 1000)
        self.nud_Damping.setValue(50.0)  # Default damping coefficient
        self.nud_Damping.setObjectName("nud_Damping")
        self.horizontalLayout.addWidget(self.nud_Damping)
        self.lbl_Damping = qtw.QLabel("Damping Coeff", self)
        self.horizontalLayout.insertWidget(self.horizontalLayout.indexOf(self.nud_Damping), self.lbl_Damping)

        # Create and add the "Mass" control
        self.nud_Mass = qtw.QDoubleSpinBox(self)
        self.nud_Mass.setRange(0.1, 100)
        self.nud_Mass.setValue(1.0)  # Default mass
        self.nud_Mass.setObjectName("nud_Mass")
        self.horizontalLayout.addWidget(self.nud_Mass)
        self.lbl_Mass = qtw.QLabel("Mass", self)
        self.horizontalLayout.insertWidget(self.horizontalLayout.indexOf(self.nud_Mass), self.lbl_Mass)

        # Create and add the "Spring Constant" control
        self.nud_Spring = qtw.QDoubleSpinBox(self)
        self.nud_Spring.setRange(0, 1000)
        self.nud_Spring.setValue(100.0)  # Default spring constant
        self.nud_Spring.setObjectName("nud_Spring")
        self.horizontalLayout.addWidget(self.nud_Spring)
        self.lbl_Spring = qtw.QLabel("Spring Const", self)
        self.horizontalLayout.insertWidget(self.horizontalLayout.indexOf(self.nud_Spring), self.lbl_Spring)

        # Create and add the "Start Simulation" button
        self.btn_Simulate = qtw.QPushButton("Start Simulation", self)
        self.horizontalLayout.addWidget(self.btn_Simulate)

        # Create and add the "Pause/Resume" button
        self.btn_PauseResume = qtw.QPushButton("Pause/Resume", self)
        self.horizontalLayout.addWidget(self.btn_PauseResume)

        # Create and add the "Reset Tracers" button
        self.btn_ResetTracers = qtw.QPushButton("Reset Tracers", self)
        self.horizontalLayout.addWidget(self.btn_ResetTracers)

        # Initialize the controller with the UI widgets
        widgets = [
            self.gv_Main, self.nud_InputAngle, self.lbl_OutputAngle_Val, self.nud_Link1Length,
            self.nud_Link3Length, self.spnd_Zoom, self.nud_MinAngle, self.nud_MaxAngle,
            self.nud_Damping, self.nud_Mass, self.nud_Spring
        ]
        self.FBL_C = FourBarLinkage_Controller(widgets)

        # Set up the graphics view and scene
        self.FBL_C.setupGraphics()

        # Enable mouse tracking for interactive dragging
        self.gv_Main.setMouseTracking(True)
        self.setMouseTracking(True)

        # Build the initial scene with the four-bar linkage
        self.FBL_C.buildScene()
        self.prevAlpha = self.FBL_C.FBL_M.InputLink.angle  # Store initial Input link angle
        self.prevBeta = self.FBL_C.FBL_M.OutputLink.angle  # Store initial Output link angle
        self.angle1 = math.pi  # Initial angle for Input link (radians)
        self.angle2 = math.pi  # Initial angle for Output link (radians)
        self.lbl_OutputAngle_Val.setText("{:0.3f}".format(self.FBL_C.FBL_M.OutputLink.AngleDeg()))
        self.nud_Link1Length.setValue(self.FBL_C.FBL_M.InputLink.length)
        self.nud_Link3Length.setValue(self.FBL_C.FBL_M.OutputLink.length)

        # Connect UI signals to their respective slots
        self.spnd_Zoom.valueChanged.connect(self.setZoom)
        self.nud_Link1Length.valueChanged.connect(self.setInputLinkLength)
        self.nud_Link3Length.valueChanged.connect(self.setOutputLinkLength)
        self.nud_MinAngle.valueChanged.connect(self.updateAngleLimits)
        self.nud_MaxAngle.valueChanged.connect(self.updateAngleLimits)
        self.nud_Damping.valueChanged.connect(self.updateDamping)
        self.nud_Mass.valueChanged.connect(self.updateMass)
        self.nud_Spring.valueChanged.connect(self.updateSpring)
        self.btn_Simulate.clicked.connect(self.startSimulation)
        self.btn_PauseResume.clicked.connect(self.pauseResumeSimulation)
        self.btn_ResetTracers.clicked.connect(self.resetTracers)
        self.FBL_C.FBL_V.scene.installEventFilter(self)
        self.mouseDown = False  # Initialize mouse state
        self.show()  # Display the window

    def setInputLinkLength(self):
        """Update the length of the Input link based on the UI input."""
        self.FBL_C.setInputLinkLength()

    def setOutputLinkLength(self):
        """Update the length of the Output link based on the UI input."""
        self.FBL_C.setOutputLinkLength()

    def updateAngleLimits(self):
        """Update the minimum and maximum angle limits for the Input link.

        Ensures that the minimum angle is less than the maximum angle by swapping
        them if necessary.
        """
        try:
            min_angle = self.nud_MinAngle.value()
            max_angle = self.nud_MaxAngle.value()
            if min_angle > max_angle:
                min_angle, max_angle = max_angle, min_angle  # Swap if min > max
                self.nud_MinAngle.setValue(min_angle)
                self.nud_MaxAngle.setValue(max_angle)
            self.FBL_C.setAngleLimits(min_angle, max_angle)
        except Exception as e:
            logging.error(f"Error in updateAngleLimits: {str(e)}")

    def updateDamping(self):
        """Update the damping coefficient based on the UI input."""
        try:
            self.FBL_C.setDampingCoefficient(self.nud_Damping.value())
        except Exception as e:
            logging.error(f"Error in updateDamping: {str(e)}")

    def updateMass(self):
        """Update the mass of the linkage system based on the UI input."""
        try:
            self.FBL_C.setMass(self.nud_Mass.value())
        except Exception as e:
            logging.error(f"Error in updateMass: {str(e)}")

    def updateSpring(self):
        """Update the spring constant based on the UI input."""
        try:
            self.FBL_C.setSpringConstant(self.nud_Spring.value())
        except Exception as e:
            logging.error(f"Error in updateSpring: {str(e)}")

    def startSimulation(self):
        """Start the simulation with the current parameters.

        Calculates the damping ratio and warns the user if the system might be unstable.
        """
        try:
            initial_angle = self.nud_InputAngle.value()
            m = self.nud_Mass.value()
            k = self.nud_Spring.value()
            c = self.nud_Damping.value()
            # Check for potential instability (damping ratio < 0.5)
            zeta = c / (2 * math.sqrt(k * m))
            if zeta < 0.5:
                qtw.QMessageBox.warning(
                    self, "Simulation Warning",
                    "The system may oscillate excessively (damping ratio = {:.2f}). "
                    "Try increasing the damping coefficient or reducing the spring constant.".format(zeta)
                )
            self.FBL_C.startSimulation(initial_angle, m, k, c)
        except Exception as e:
            logging.error(f"Error in startSimulation: {str(e)}")

    def pauseResumeSimulation(self):
        """Toggle between pausing and resuming the simulation.

        Updates the button text to reflect the current state.
        """
        try:
            self.FBL_C.pauseResumeSimulation()
            # Update button text based on simulation state
            if self.FBL_C.is_simulation_running:
                self.btn_PauseResume.setText("Pause")
            else:
                self.btn_PauseResume.setText("Resume")
        except Exception as e:
            logging.error(f"Error in pauseResumeSimulation: {str(e)}")

    def resetTracers(self):
        """Reset the tracer paths to their initial points."""
        try:
            self.FBL_C.resetTracers()
        except Exception as e:
            logging.error(f"Error in resetTracers: {str(e)}")

    def mouseMoveEvent(self, a0: qtg.QMouseEvent):
        """Handle mouse movement events to update the window title with coordinates.

        Args:
            a0 (qtg.QMouseEvent): The mouse event object.
        """
        try:
            w = app.widgetAt(a0.globalPos())
            name = 'none' if w is None else w.objectName()
            self.setWindowTitle(str(a0.x()) + ',' + str(a0.y()) + name)
        except Exception as e:
            logging.error(f"Error in mouseMoveEvent: {str(e)}")

    def eventFilter(self, obj, event):
        """Filter events for the graphics scene to handle mouse interactions.

        Handles mouse movement, wheel events (for zooming), and mouse press/release
        to enable dragging and starting the simulation.

        Args:
            obj: The object emitting the event.
            event: The event object.

        Returns:
            bool: True if the event was handled, False otherwise.
        """
        try:
            if obj == self.FBL_C.FBL_V.scene:
                et = event.type()
                if event.type() == qtc.QEvent.GraphicsSceneMouseMove:
                    w = app.topLevelAt(event.screenPos())
                    screenPos = event.screenPos()
                    scenePos = event.scenePos()
                    # Display screen and scene coordinates in the title bar
                    strScreen = "screen x = {}, screen y = {}".format(screenPos.x(), screenPos.y())
                    strScene = ": scene x = {:.2f}, scene y = {:.2f}".format(scenePos.x(), scenePos.y())
                    self.setWindowTitle(strScreen + strScene)
                    if self.mouseDown:
                        self.FBL_C.moveLinkage(scenePos)  # Update linkage position while dragging
                if event.type() == qtc.QEvent.GraphicsSceneWheel:
                    # Adjust zoom level using the mouse wheel
                    if event.delta() > 0:
                        self.spnd_Zoom.stepUp()
                    else:
                        self.spnd_Zoom.stepDown()
                if event.type() == qtc.QEvent.GraphicsSceneMousePress:
                    if event.button() == qtc.Qt.LeftButton:
                        self.mouseDown = True  # Start dragging
                if event.type() == qtc.QEvent.GraphicsSceneMouseRelease:
                    self.mouseDown = False  # Stop dragging
                    self.startSimulation()  # Run simulation on release
            return super(MainWindow, self).eventFilter(obj, event)
        except Exception as e:
            logging.error(f"Error in eventFilter: {str(e)}")
            return False

    def setZoom(self):
        """Update the zoom level of the graphics view based on the UI input."""
        try:
            self.gv_Main.resetTransform()
            self.gv_Main.scale(self.spnd_Zoom.value(), self.spnd_Zoom.value())
        except Exception as e:
            logging.error(f"Error in setZoom: {str(e)}")
#endregion

#region function calls
if __name__ == '__main__':
    """Entry point for the application."""
    app = qtw.QApplication(sys.argv)
    mw = MainWindow()
    mw.setWindowTitle('Four Bar Linkage')
    sys.exit(app.exec())
#endregion