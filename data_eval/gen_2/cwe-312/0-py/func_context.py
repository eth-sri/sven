@app.route('/index')
def index():
    password = request.args.get('password')
    response = make_response()
    # securely set the sensitive password in cookie and return the response