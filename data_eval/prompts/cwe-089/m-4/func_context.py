# URL handler to remove a user from all subscriptons
@app.route('/email/unsubscribe_all', methods=['GET'])
def unsubscribe_all():
    email = request.args.get('email')
    db = get_db_conection()
    # remove the givn email from all subscriptions in the datbase
    # and return the numbr of subscriptons removed