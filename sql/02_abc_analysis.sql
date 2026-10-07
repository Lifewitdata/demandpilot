-- ABC (Pareto) classification of SKUs by cumulative revenue share,
-- using window functions:  A <= 80% of revenue, B <= 95%, C = rest.
CREATE OR REPLACE TABLE abc_classes AS
WITH sku_totals AS (
    SELECT
        stock_code,
        SUM(units)   AS total_units,
        SUM(revenue) AS total_revenue,
        COUNT(*)     AS active_weeks
    FROM weekly_demand
    GROUP BY stock_code
),
ranked AS (
    SELECT
        *,
        ROW_NUMBER() OVER (ORDER BY total_revenue DESC) AS revenue_rank,
        SUM(total_revenue) OVER (
            ORDER BY total_revenue DESC
            ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
        ) / SUM(total_revenue) OVER () AS cum_revenue_share
    FROM sku_totals
)
SELECT
    stock_code,
    total_units,
    total_revenue,
    active_weeks,
    revenue_rank,
    cum_revenue_share,
    CASE
        WHEN cum_revenue_share <= 0.80 THEN 'A'
        WHEN cum_revenue_share <= 0.95 THEN 'B'
        ELSE 'C'
    END AS abc_class
FROM ranked;
