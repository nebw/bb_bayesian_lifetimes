import pandas as pd
import psycopg2


default_query_prefix = """
SET geqo_effort to 10;
SET max_parallel_workers_per_gather TO 16;
SET temp_buffers to "32GB";
SET work_mem to "1GB";
"""


def get_detection_counts(table_name, connect_str, query_prefix=default_query_prefix):
    with psycopg2.connect(connect_str) as conn:
        detections = pd.read_sql_query(
            """
            {}

            SELECT EXTRACT(DOY FROM timestamp) as doy, bee_id, COUNT(*)
            FROM {}
            GROUP BY doy, bee_id
            """.format(
                query_prefix, table_name
            ),
            conn,
        )

    return detections
