import pandas as pd
import plotly.express as px

def prediction_chart(predictions):

    if not predictions:
        return None

    df = pd.DataFrame(predictions)

    fig = px.bar(
        df,
        x="condition",
        y="confidence",
        text="confidence",
    )

    fig.update_layout(
        height=420,
        template="plotly_white",
    )

    return fig