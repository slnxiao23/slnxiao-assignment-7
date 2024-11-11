from flask import Flask, render_template, request, url_for, session
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import scipy.stats as stats

app = Flask(__name__)
app.secret_key = "apple"

def generate_data(N, mu, beta0, beta1, sigma2, S):
    X = np.random.rand(N)
    Y = beta0 + beta1 * X + mu + np.sqrt(sigma2) * np.random.randn(N)
    X_reshaped = X.reshape(-1, 1)
    model = LinearRegression().fit(X_reshaped, Y)
    slope = model.coef_[0]
    intercept = model.intercept_

    plt.figure(figsize=(8, 6))
    plt.scatter(X, Y, color='blue', label='Data Points')
    plt.plot(X, model.predict(X_reshaped), color='red', label=f'Regression Line: Y = {slope:.2f}X + {intercept:.2f}')
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title(f"Scatter Plot with Regression Line\nSlope: {slope:.2f}, Intercept: {intercept:.2f}")
    plt.legend()
    plot1_path = "static/plot1.png"
    plt.savefig(plot1_path)
    plt.close()

    slopes = []
    intercepts = []
    for _ in range(S):
        X_sim = np.random.rand(N)
        Y_sim = beta0 + beta1 * X_sim + mu + np.sqrt(sigma2) * np.random.randn(N)
        X_sim_reshaped = X_sim.reshape(-1, 1)
        sim_model = LinearRegression().fit(X_sim_reshaped, Y_sim)
        slopes.append(sim_model.coef_[0])
        intercepts.append(sim_model.intercept_)

    plt.figure(figsize=(10, 5))
    plt.hist(slopes, bins=20, alpha=0.5, color="blue", label="Simulated Slopes")
    plt.hist(intercepts, bins=20, alpha=0.5, color="orange", label="Simulated Intercepts")
    plt.axvline(slope, color="blue", linestyle="--", linewidth=1, label=f"Observed Slope: {slope:.2f}")
    plt.axvline(intercept, color="orange", linestyle="--", linewidth=1, label=f"Observed Intercept: {intercept:.2f}")
    plt.title("Histogram of Simulated Slopes and Intercepts")
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.legend()
    plot2_path = "static/plot2.png"
    plt.savefig(plot2_path)
    plt.close()

    return X, Y, slope, intercept, plot1_path, plot2_path, slopes, intercepts

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        N = int(request.form["N"])
        mu = float(request.form["mu"])
        sigma2 = float(request.form["sigma2"])
        beta0 = float(request.form["beta0"])
        beta1 = float(request.form["beta1"])
        S = int(request.form["S"])

        X, Y, slope, intercept, plot1, plot2, slopes, intercepts = generate_data(N, mu, beta0, beta1, sigma2, S)

        session["X"] = X.tolist()
        session["Y"] = Y.tolist()
        session["slope"] = slope
        session["intercept"] = intercept
        session["slopes"] = slopes
        session["intercepts"] = intercepts
        session["N"] = N
        session["mu"] = mu
        session["sigma2"] = sigma2
        session["beta0"] = beta0
        session["beta1"] = beta1
        session["S"] = S
        session["plot1"] = plot1
        session["plot2"] = plot2
        session["parameter"] = "slope"
    else:
        session.clear()

    return render_template("index.html",
                           plot1=session.get("plot1"),
                           plot2=session.get("plot2"))

@app.route("/hypothesis_test", methods=["POST"])
def hypothesis_test():
    slope = float(session.get("slope"))
    intercept = float(session.get("intercept"))
    slopes = session.get("slopes")
    intercepts = session.get("intercepts")
    beta0 = float(session.get("beta0"))
    beta1 = float(session.get("beta1"))
    parameter = request.form.get("parameter")
    test_type = request.form.get("test_type")

    if parameter == "slope":
        simulated_stats = np.array(slopes)
        observed_stat = slope
        hypothesized_value = beta1
    else:
        simulated_stats = np.array(intercepts)
        observed_stat = intercept
        hypothesized_value = beta0

    if test_type == ">":
        p_value = np.mean(simulated_stats >= observed_stat)
    elif test_type == "<":
        p_value = np.mean(simulated_stats <= observed_stat)
    else:
        p_value = np.mean(np.abs(simulated_stats - hypothesized_value) >= np.abs(observed_stat - hypothesized_value))

    fun_message = "Rare event encountered!" if p_value <= 0.0001 else None

    plt.figure(figsize=(8, 6))
    plt.hist(simulated_stats, bins=20, color="skyblue", alpha=0.7, label="Simulated Statistics")
    plt.axvline(observed_stat, color="red", linestyle="--", label=f"Observed {parameter.capitalize()}: {observed_stat:.4f}")
    plt.axvline(hypothesized_value, color="blue", linestyle="--", label=f"Hypothesized {parameter.capitalize()} (H₀): {hypothesized_value}")
    plt.xlabel(parameter.capitalize())
    plt.ylabel("Frequency")
    plt.legend()
    plt.title(f"Hypothesis Test for {parameter.capitalize()}")
    plot3_path = "static/plot3.png"
    plt.savefig(plot3_path)
    plt.close()

    session["plot3"] = plot3_path
    session["p_value"] = p_value
    session["fun_message"] = fun_message
    session["parameter"] = parameter
    session["observed_stat"] = observed_stat
    session["hypothesized_value"] = hypothesized_value

    return render_template("index.html",
                           plot1=session.get("plot1"),
                           plot2=session.get("plot2"),
                           plot3=session.get("plot3"),
                           p_value=p_value,
                           fun_message=fun_message,
                           parameter=parameter,
                           observed_stat=observed_stat,
                           hypothesized_value=hypothesized_value)

@app.route("/confidence_interval", methods=["POST"])
def confidence_interval():
    beta0 = float(session.get("beta0"))
    beta1 = float(session.get("beta1"))
    slopes = session.get("slopes")
    intercepts = session.get("intercepts")
    parameter = request.form.get("parameter")
    confidence_level = float(request.form.get("confidence_level"))

    if parameter == "slope":
        estimates = np.array(slopes)
        true_param = beta1
    else:
        estimates = np.array(intercepts)
        true_param = beta0

    mean_estimate = np.mean(estimates)
    std_estimate = np.std(estimates, ddof=1)
    margin_of_error = stats.t.ppf((1 + confidence_level / 100) / 2, len(estimates) - 1) * std_estimate
    ci_lower = mean_estimate - margin_of_error
    ci_upper = mean_estimate + margin_of_error
    includes_true = bool(ci_lower <= true_param <= ci_upper)

    plt.figure(figsize=(8, 6))
    plt.scatter(estimates, [1] * len(estimates), color="gray", alpha=0.5, label="Simulated Estimates")
    plt.plot([mean_estimate], [1], marker="o", color="blue", label="Mean Estimate")
    plt.hlines(y=1, xmin=ci_lower, xmax=ci_upper, color="blue", label=f"{confidence_level}% Confidence Interval")
    plt.axvline(true_param, color="green", linestyle="--", label="True Parameter")
    plt.xlabel(f"{parameter.capitalize()} Estimate")
    plt.title(f"{confidence_level}% Confidence Interval for {parameter.capitalize()}")
    plt.legend()
    plot4_path = "static/plot4.png"
    plt.savefig(plot4_path)
    plt.close()

    session["plot4"] = plot4_path
    session["mean_estimate"] = mean_estimate
    session["ci_lower"] = ci_lower
    session["ci_upper"] = ci_upper
    session["includes_true"] = includes_true
    session["confidence_level"] = confidence_level
    session["parameter"] = parameter

    return render_template("index.html",
                           plot1=session.get("plot1"),
                           plot2=session.get("plot2"),
                           plot3=session.get("plot3"),
                           plot4=session.get("plot4"),
                           mean_estimate=mean_estimate,
                           ci_lower=ci_lower,
                           ci_upper=ci_upper,
                           includes_true=includes_true,
                           confidence_level=confidence_level,
                           parameter=parameter)

if __name__ == "__main__":
    app.run(debug=True)
