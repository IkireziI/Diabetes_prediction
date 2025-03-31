from flask import Flask, render_template, url_for

app = Flask(__name__)

@app.route('/')
def test_route():
    return render_template('test.html')

if __name__ == '__main__':
    app.run(debug=True, use_reloader=True, host='0.0.0.0', port=5000)