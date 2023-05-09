@app.route('/email/unsubscribe_all', methods=['GET'])
def unsubscribe_all():
    """URL handler to remove a user from all subscriptions"""
    email = request.args.get('email')
    db = get_db_connection()
    """remove the given email from all subscriptions in the database
    and return the number of subscriptions removed"""