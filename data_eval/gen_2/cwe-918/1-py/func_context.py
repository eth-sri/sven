@app.route("/visit")
def visit():
    user_id = request.args.get('user_id')
    # use user_id as a part of an url to get user info, make a request to the url, and return the response