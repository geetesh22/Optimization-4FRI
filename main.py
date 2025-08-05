import streamlit as st
import pulp
import pandas as pd
import matplotlib.pyplot as plt

###############################################################################
# CONSTANTS
###############################################################################
EQUIP_LIFE_YEARS = 30  # years of depreciation for all equipment

###############################################################################
# SOLVER LOGIC
###############################################################################

def solve_biomass_model(
        # Feedstock availability
        slash_avail,
        woodchips_avail,
        include_woodchips,

        # The list of products to be optimized
        selected_products,

        # Maximum volume (capacity) for each product p
        max_volume,

        # Per‑product variable & capital costs
        processing_cost,
        depreciation_per_ton,

        # For each product p:
        #   slash_harvest_cost[p], slash_transport_cost[p], slash_wood_cost[p], slash_carbon_credit[p]
        #   woodchips_harvest_cost[p], woodchips_transport_cost[p], woodchips_wood_cost[p], woodchips_carbon_credit[p]
        #   market_price[p], max_deliv_cost[p]
        # We also have a regulatory factor that inflates the compliance‑related portion of delivered cost
        slash_harvest_cost,
        slash_transport_cost,
        slash_wood_cost,
        slash_carbon_credit,
        woodchips_harvest_cost,
        woodchips_transport_cost,
        woodchips_wood_cost,
        woodchips_carbon_credit,
        market_price,
        max_deliv_cost,
        reg_factor
):
    """
    Builds a multi‑feedstock (Slash, Woodchips) × multi‑product model
    for only the products in selected_products.

    DeliveredCost = (HarvestCost + TransportCost + WoodCost) × (1 + reg_factor)
                    + ProcessingCost + DepreciationCost.
    NetMargin     = (Price + CarbonCredit) − DeliveredCost.

    Returns:
      status_str        : solver status (string, e.g. "Optimal")
      df_details        : DataFrame with allocated tons and financial details
      total_net_revenue : objective value (float)
    """

    feedstocks = ["slash"]
    if include_woodchips:
        feedstocks.append("woodchips")

    # Create the LP model
    model = pulp.LpProblem("4FRI_MultiProduct_LightMode", sense=pulp.LpMaximize)

    # Decision Variables: Q[(f, p)] in green tons
    Q = {}
    for p in selected_products:
        for f in feedstocks:
            Q[(f, p)] = pulp.LpVariable(f"Q_{f}_{p}", lowBound=0, cat=pulp.LpContinuous)

    # OBJECTIVE: sum of net margin across all feedstock‑product combos
    obj_terms = []
    for p in selected_products:
        for f in feedstocks:
            # compliance‑sensitive cost = harvest + transport + wood
            if f == "slash":
                base_cost = (slash_harvest_cost[p] +
                             slash_transport_cost[p] +
                             slash_wood_cost[p])
                carbon = slash_carbon_credit[p]
            else:  # woodchips
                base_cost = (woodchips_harvest_cost[p] +
                             woodchips_transport_cost[p] +
                             woodchips_wood_cost[p])
                carbon = woodchips_carbon_credit[p]

            delivered_cost = base_cost * (1 + reg_factor) \
                             + processing_cost[p] + depreciation_per_ton[p]
            revenue_per_ton = market_price[p] + carbon
            net_margin = revenue_per_ton - delivered_cost

            obj_terms.append(net_margin * Q[(f, p)])

    model += pulp.lpSum(obj_terms), "Total_Net_Revenue"

    # -------------------- CONSTRAINTS --------------------
    # 1) Slash availability
    model += pulp.lpSum(
        [Q[("slash", p)] for p in selected_products if ("slash", p) in Q]
    ) <= slash_avail, "SlashAvail"

    # 2) Woodchips availability
    if include_woodchips:
        model += pulp.lpSum(
            [Q[("woodchips", p)] for p in selected_products if ("woodchips", p) in Q]
        ) <= woodchips_avail, "WoodchipsAvail"

    # 3) Max Delivered Cost constraints (per product p)
    #    Σ(deliveredCost_tons) − max_deliv_cost[p] × Σ(tons) ≤ 0
    for p in selected_products:
        sum_cost_expr = []
        sum_tons_expr = []
        for f in feedstocks:
            if (f, p) not in Q:
                continue
            if f == "slash":
                base_cost = (slash_harvest_cost[p] +
                             slash_transport_cost[p] +
                             slash_wood_cost[p])
            else:
                base_cost = (woodchips_harvest_cost[p] +
                             woodchips_transport_cost[p] +
                             woodchips_wood_cost[p])

            c_deliver = base_cost * (1 + reg_factor) + processing_cost[p] + depreciation_per_ton[p]
            sum_cost_expr.append(c_deliver * Q[(f, p)])
            sum_tons_expr.append(Q[(f, p)])

        model += (
            pulp.lpSum(sum_cost_expr) <= max_deliv_cost[p] * pulp.lpSum(sum_tons_expr)
        ), f"MaxDeliveredCost_{p}"

    # 4) Max Volume constraints: Σ_f Q[f, p] ≤ max_volume[p]
    for p in selected_products:
        model += pulp.lpSum(
            [Q[(f, p)] for f in feedstocks if (f, p) in Q]
        ) <= max_volume[p], f"MaxVolume_{p}"

    # -------------------- SOLVE --------------------
    status = model.solve(pulp.PULP_CBC_CMD(msg=0))
    status_str = pulp.LpStatus[status]
    total_net_revenue = pulp.value(model.objective) if model.objective else 0.0

    # -------------------- DETAIL TABLE --------------------
    rows = []
    for p in selected_products:
        for f in feedstocks:
            if (f, p) not in Q:
                continue
            allocated = Q[(f, p)].varValue or 0.0
            if allocated < 1e-6:
                continue

            # Recompute delivered cost & revenue details for each row
            if f == "slash":
                base_cost = (slash_harvest_cost[p] +
                             slash_transport_cost[p] +
                             slash_wood_cost[p])
                carbon = slash_carbon_credit[p]
            else:
                base_cost = (woodchips_harvest_cost[p] +
                             woodchips_transport_cost[p] +
                             woodchips_wood_cost[p])
                carbon = woodchips_carbon_credit[p]

            delivered_cost_per_ton = base_cost * (1 + reg_factor) + processing_cost[p] + depreciation_per_ton[p]
            revenue_per_ton = market_price[p] + carbon
            net_margin_per_ton = revenue_per_ton - delivered_cost_per_ton

            rows.append({
                "Product": p,
                "Feedstock": f,
                "Allocated (green tons)": round(allocated, 2),
                "DeliveredCost ($/gt)": round(delivered_cost_per_ton, 2),
                "Revenue ($/gt)": round(revenue_per_ton, 2),
                "Net Margin ($/gt)": round(net_margin_per_ton, 2),
                "Processing Cost ($/gt)": round(processing_cost[p], 2),
                "Depreciation ($/gt)": round(depreciation_per_ton[p], 2),
                "Total DeliveredCost ($)": round(delivered_cost_per_ton * allocated, 2),
                "Total Revenue ($)": round(revenue_per_ton * allocated, 2),
                "Total Net Margin ($)": round(net_margin_per_ton * allocated, 2)
            })

    df_details = pd.DataFrame(rows)
    return status_str, df_details, total_net_revenue

###############################################################################
# STREAMLIT APP (LIGHT MODE) - MAIN
###############################################################################

def main():
    # Page config
    st.set_page_config(page_title="Biomass Optimization (Light Mode)", layout="wide")

    st.title("Multi‑Product Biomass Optimization")
    st.write("""
    **DeliveredCost** = (HarvestCost + TransportCost + WoodCost) × (1 + Regulatory Factor)  
    **+ ProcessingCost + DepreciationCost**  
    **Net Margin** = (MarketPrice + CarbonCredit) − DeliveredCost  
    You can also set a **Max Volume** for each product and select which products to include.
    """)

    st.markdown("---")
    st.markdown(
        "**Instructions**: Use the **sidebar** to set feedstock availability, costs, "
        "and constraints. Then click **Run Optimization**."
    )

    # -------------------- SIDEBAR --------------------
    st.sidebar.header("Feedstock Availability")
    slash_avail = st.sidebar.number_input("Slash Availability (green tons)", value=20000.0, min_value=0.0)

    include_woodchips = st.sidebar.checkbox("Include Woodchips?", value=True)
    woodchips_avail = 0.0
    if include_woodchips:
        woodchips_avail = st.sidebar.number_input("Woodchips Availability (green tons)", value=15000.0, min_value=0.0)

    # Regulatory Factor – cost of compliance multiplier
    reg_factor = st.sidebar.slider("Regulatory Factor (cost‑of‑compliance multiplier)", 0.0, 1.0, 0.2, 0.01)

    st.sidebar.header("Products to Optimize")
    all_products = ["Biochar", "RNG", "eMethanol"]
    selected_products = []
    for p in all_products:
        use_p = st.sidebar.checkbox(f"Use {p}?", value=True)
        if use_p:
            selected_products.append(p)

    if not selected_products:
        st.warning("No products selected. Please enable at least one product in the sidebar.")
        return

    # -------------------- COLLECT COST PARAMETERS --------------------
    slash_harvest_cost = {}
    slash_transport_cost = {}
    slash_wood_cost = {}
    slash_carbon_credit = {}

    woodchips_harvest_cost = {}
    woodchips_transport_cost = {}
    woodchips_wood_cost = {}
    woodchips_carbon_credit = {}

    market_price = {}
    max_deliv_cost = {}
    max_volume = {}

    processing_cost = {}
    capex_total = {}
    depreciation_per_ton = {}

    for p in selected_products:
        with st.sidebar.expander(f"{p} Settings"):
            # ---------------- MARKET ----------------
            mp = st.number_input(f"{p} Market Price [$/gt]",
                                 value=150.0 if p == "Biochar" else (180.0 if p == "RNG" else 220.0),
                                 min_value=0.0)
            market_price[p] = mp

            # ---------------- CONSTRAINTS ----------------
            mdc = st.number_input(f"{p} Max Delivered Cost [$/gt]", value=130.0, min_value=0.0)
            max_deliv_cost[p] = mdc

            mv = st.number_input(f"{p} Max Volume (green tons)", value=999999.0, min_value=1.0)
            max_volume[p] = mv

            # ---------------- SLASH COSTS ----------------
            st.write(f"**Slash → {p}**")
            s_h = st.number_input(f"Slash Harvest: {p} [$/gt]", value=25.0, min_value=0.0)
            s_t = st.number_input(f"Slash Transport: {p} [$/gt]", value=30.0, min_value=0.0)
            s_w = st.number_input(f"Slash Wood Cost: {p} [$/gt]", value=0.0, min_value=0.0)
            s_c = st.number_input(f"Slash CarbonCredit: {p} [$/gt]", value=15.0 if p == "Biochar" else 0.0, min_value=0.0)

            slash_harvest_cost[p] = s_h
            slash_transport_cost[p] = s_t
            slash_wood_cost[p] = s_w
            slash_carbon_credit[p] = s_c

            # ---------------- WOODCHIPS COSTS ----------------
            st.write(f"**Woodchips → {p}**")
            w_h = st.number_input(f"Woodchips Harvest: {p} [$/gt]", value=20.0, min_value=0.0)
            w_t = st.number_input(f"Woodchips Transport: {p} [$/gt]", value=25.0, min_value=0.0)
            w_w = st.number_input(f"Woodchips Wood Cost: {p} [$/gt]", value=0.0, min_value=0.0)
            w_c = st.number_input(f"Woodchips CarbonCredit: {p} [$/gt]", value=0.0, min_value=0.0)

            woodchips_harvest_cost[p] = w_h
            woodchips_transport_cost[p] = w_t
            woodchips_wood_cost[p] = w_w
            woodchips_carbon_credit[p] = w_c

            # ---------------- PROCESSING & CAPEX ----------------
            st.markdown("---")
            pc = st.number_input(f"Processing OPEX: {p} [$/gt]", value=10.0, min_value=0.0)
            processing_cost[p] = pc

            cap = st.number_input(f"Total CAPEX for {p} [$]", value=0.0, min_value=0.0, step=1000.0)
            capex_total[p] = cap

            # Compute straight‑line depreciation per ton using max_volume as expected annual capacity
            if mv > 0:
                depreciation_per_ton[p] = cap / EQUIP_LIFE_YEARS / mv
            else:
                depreciation_per_ton[p] = 0.0

    # -------------------- RUN OPTIMIZATION --------------------
    if st.sidebar.button("Run Optimization"):
        status_str, df_details, total_net_rev = solve_biomass_model(
            slash_avail,
            woodchips_avail,
            include_woodchips,
            selected_products,
            max_volume,
            processing_cost,
            depreciation_per_ton,
            slash_harvest_cost,
            slash_transport_cost,
            slash_wood_cost,
            slash_carbon_credit,
            woodchips_harvest_cost,
            woodchips_transport_cost,
            woodchips_wood_cost,
            woodchips_carbon_credit,
            market_price,
            max_deliv_cost,
            reg_factor
        )

        # -------------------- RESULTS --------------------
        st.subheader("Results")
        st.write(f"**Solver Status**: {status_str}")
        st.write(f"**Total Net Revenue**: ${round(total_net_rev, 2):,}")

        if df_details.empty:
            st.warning("No biomass allocated (all zero). Possibly your costs are too high or constraints too strict.")
        else:
            st.write("**Allocation & Financial Details** (non‑zero green tons):")
            st.dataframe(df_details.style.format({
                "Allocated (green tons)": "{:.2f}",
                "DeliveredCost ($/gt)": "{:.2f}",
                "Revenue ($/gt)": "{:.2f}",
                "Net Margin ($/gt)": "{:.2f}",
                "Processing Cost ($/gt)": "{:.2f}",
                "Depreciation ($/gt)": "{:.2f}",
                "Total DeliveredCost ($)": "{:.2f}",
                "Total Revenue ($)": "{:.2f}",
                "Total Net Margin ($)": "{:.2f}"
            }))

            # Bar chart – allocation by product & feedstock
            pivoted = df_details.pivot(index="Product", columns="Feedstock", values="Allocated (green tons)").fillna(0)
            fig, ax = plt.subplots()
            pivoted.plot(kind="bar", ax=ax)
            ax.set_title("Optimal Allocation (green tons)")
            ax.set_xlabel("Product")
            ax.set_ylabel("Green Tons")
            st.pyplot(fig)

    # -------------------- LEGEND --------------------
    st.markdown("---")
    st.markdown("### Legend / Explanation")
    st.markdown("""
    - **Regulatory Factor**: Cost‑of‑compliance multiplier applied to harvest, transport & wood costs.
    - **Processing Cost**: Variable OPEX per green ton processed (e.g. utilities, labor, catalysts).
    - **Depreciation**: Straight‑line CAPEX recovery → (Total CAPEX / 30 yrs / Max Volume) in $/gt.
    - **Delivered Cost** = (Harvest + Transport + Wood) × (1 + RegFactor) + Processing + Depreciation.
    - **Net Margin** = (Price + CarbonCredit) − DeliveredCost.
    - **Max Delivered Cost**: Ensures average delivered cost for each product ≤ user‑set limit.
    - **Max Volume**: Total annual tonnage for each product cannot exceed this limit.
    - **Slash / Woodchips**: Two feedstocks, each with separate availability and cost structures.
    - **Market Price**: $/green ton of product (ex‑plant gate).
    - **Carbon Credit**: Additional revenue for that feedstock route (often higher for slash).
    - **Q_{f,p}**: Decision variable – how many green tons of feedstock *f* go to product *p*.
    - **(Total Revenue / Net Margin)** = per‑ton value × allocated tons.
    """)


if __name__ == "__main__":
    main()
