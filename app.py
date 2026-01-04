from flask import Flask, request, render_template

from src.pipeline.predict_pipeline import CustomData, PredictPipeline



application = Flask(__name__)
app = application

# Routes

@app.route("/", methods=['GET'])
def getHome():
    return render_template("index.html")

@app.route("/predictdata", methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template("home.html")
    else:
        gender = request.form.get("gender")
        ethnicity = request.form.get("ethnicity")
        parental_level_of_education = request.form.get("parental_level_of_education")
        lunch = request.form.get("lunch")
        test_preparation_course = request.form.get("test_preparation_course")
        reading_score = request.form.get("reading_score")
        writing_score = request.form.get("writing_score")
        
        data_obj = CustomData(gender, ethnicity, parental_level_of_education, lunch, test_preparation_course, reading_score, writing_score)
        
        pred_data = data_obj.get_custom_data()
        predict_pipeline_obj = PredictPipeline()

        results = predict_pipeline_obj.predict_data(pred_data)

        return render_template("home.html", results=results[0])

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5050, debug=True)
