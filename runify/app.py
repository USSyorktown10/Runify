from flask import Flask, render_template, request, jsonify, redirect, url_for
import os
import sqlite3
import json

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/Runify.png')
def serve_image():
    return app.send_static_file('Runify.png')

@app.route('/bg-favicon.png')
def serve_icon():
    return app.send_static_file('bg-favicon.png')

# Sample data mimicking a database lookup
user_data = {
    'firstname': 'Julian'
}

stats_data = {
    'weekly_mileage': 25.5,
    'last_distance': 5.2,
    'last_pace': '8:15 min/mi',
    'avg_hr': '135 bpm'
}

recent_activities_data = [
    {
        'name': 'Morning Run',
        'distance': 4.1,
        'pace': '8:30',
        'time': '34:45',
        'date': 'Nov 15'
    },
    {
        'name': 'Evening Jog',
        'distance': 3.0,
        'pace': '9:00',
        'time': '27:00',
        'date': 'Nov 14'
    },
    {
        'name': 'Long Run',
        'distance': 10.0,
        'pace': '8:00',
        'time': '1:20:00',
        'date': 'Nov 12'
    }
]

insights_data = {
    'insight_message': 'Great week of consistent running! Heart rate looks stable.',
    'recommendation': 'Today: Easy 3 mile recovery run to aid muscle repair.',
    'trend': 'You have increased your weekly mileage by 15% over last week.'
}

@app.route('/dashboard')
def dashboard():
    # Pass all the necessary data structures as keyword arguments
    return render_template(
        'dashboard.html',
        user=user_data,                  # Corresponds to {{ user.firstname }}
        weekly_mileage=stats_data['weekly_mileage'],
        last_distance=stats_data['last_distance'],
        last_pace=stats_data['last_pace'],
        avg_hr=stats_data['avg_hr'],
        recent=recent_activities_data,   # Corresponds to {% for a in recent %}
        insight_message=insights_data['insight_message'],
        recommendation=insights_data['recommendation'],
        trend=insights_data['trend']
    )


@app.route('/activities')
def activities():
    return render_template('activities.html', recent=recent_activities_data)


@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        f = request.files.get('file')
        if f:
            # process files here later
            return jsonify({'ok': True, 'activity_id': 123})
        return jsonify({'ok': False}), 400
    return render_template('upload.html')


@app.route('/login')
def login():
    # login flow placeholder
    return render_template('login.html')


@app.route('/signup')
def signup():
    # signup flow placeholder
    return render_template('signup.html')


@app.route('/logout')
def logout():
    # simple logout flow placeholder
    return render_template('logout.html')


@app.route('/settings')
def settings():
    return render_template('settings.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)