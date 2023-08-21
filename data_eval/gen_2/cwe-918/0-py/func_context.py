@app.route("/visit")
def visit():
    subdomain = request.args.get('subdomain')
    # safely use subdomain in an url, make a request to the url, and return the response