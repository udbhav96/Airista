from flask import Flask, render_template, redirect, url_for, Response  # Added Response
import vp  # This is your Python script (make sure it has functions)

app = Flask(__name__)

# First page
@app.route('/')
def index():
    return render_template('index.html')

# Second page (camera)
@app.route('/camera')
def camera():
    return render_template('camera.html')

# Third page (runs Python script)
@app.route('/vp')
def vp_page():
    # This should return the template that shows the video feed
    return render_template('vp.html')  # Make sure this line exists

@app.route('/video_feed')
def video_feed():
    return Response(vp.generate_frames(), 
                  mimetype='multipart/x-mixed-replace; boundary=frame')
# Fourth page (thanks)
@app.route('/thanks')
def thanks():
    return render_template('thanks.html')

if __name__ == '__main__':
    app.run(debug=True)
