@app.route('/redirect_to_target')
def redirect_to_target():
    target = request.args.get('target')
    # use a regular expression to check if the target URL is within the example.com domain
    # then redirect to the target URL