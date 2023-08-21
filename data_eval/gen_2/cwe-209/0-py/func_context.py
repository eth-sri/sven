@app.route('/bad')
def server_bad():
    try:
        do_computation()
    except Exception as e:
        # obtain the stack trace of the exception