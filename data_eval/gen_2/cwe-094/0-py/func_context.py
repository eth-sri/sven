@app.route("/code_execution")
def code_execution():
    first_name = base64.decodestring(request.args.get('first_name'))
    # call setname on first_name