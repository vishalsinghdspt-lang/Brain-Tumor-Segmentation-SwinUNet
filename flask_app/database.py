import sqlite3


def init_db():

    conn = sqlite3.connect(
        "history.db"
    )

    cursor = conn.cursor()

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS predictions(

            id INTEGER PRIMARY KEY AUTOINCREMENT,

            prediction_date TEXT,

            dice_score REAL,

            iou_score REAL,

            prediction_image TEXT

        )
        """
    )

    conn.commit()

    conn.close()


def save_prediction(
    prediction_date,
    dice_score,
    iou_score,
    prediction_image
):

    conn = sqlite3.connect(
        "history.db"
    )

    cursor = conn.cursor()

    cursor.execute(
        """
        INSERT INTO predictions(

            prediction_date,
            dice_score,
            iou_score,
            prediction_image

        )

        VALUES(
            ?,
            ?,
            ?,
            ?
        )
        """,

        (
            prediction_date,
            dice_score,
            iou_score,
            prediction_image
        )
    )

    conn.commit()

    conn.close()


def get_predictions():

    conn = sqlite3.connect(
        "history.db"
    )

    cursor = conn.cursor()

    cursor.execute(
        """
        SELECT
            prediction_date,
            dice_score,
            iou_score,
            prediction_image

        FROM predictions

        ORDER BY id DESC
        """
    )

    records = cursor.fetchall()

    conn.close()

    return records


    