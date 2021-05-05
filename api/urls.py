from api.services import predictor


def app_routes(app):
    app.add_url_rule(rule="/", view_func=predictor.home, methods=["GET"])
    app.add_url_rule(rule="/predict", view_func=predictor.predict, methods=["POST"])
