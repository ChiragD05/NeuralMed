def imaging_prediction_chart(predictions):

    import pandas as pd
    import plotly.express as px

    df = pd.DataFrame(predictions)

    fig = px.bar(
        df,
        x="class",
        y="confidence",
        text="confidence",
    )

    fig.update_layout(
        height=420,
        template="plotly_dark",
    )

    return fig