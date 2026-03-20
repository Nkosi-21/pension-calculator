import streamlit as st
import numpy as np
import joblib
import os
import warnings

warnings.filterwarnings('ignore')

st.set_page_config(page_title="Pension Calculator", layout="wide")

st.markdown("""
<style>
    .benefit-box {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
        border: 2px solid #1a3a5c;
        margin-bottom: 25px;
    }
    .benefit-value {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1a3a5c;
    }
    .metric-box {
        background: linear-gradient(135deg, #1a3a5c, #2e6da4);
        border-radius: 10px;
        padding: 15px;
        text-align: center;
        color: white;
    }

</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model():
    MODEL_PATH = "Final_Pension_Model.pkl"
    if not os.path.exists(MODEL_PATH):
        st.error(f"Model file not found at: {MODEL_PATH}")
        return None
    model = joblib.load(MODEL_PATH)
    return model


def predict_benefit(model, gender, entry_age, exit_age, duration, salary, prev_bal, contrib, inv, inf):
    real_return = inv - inf
    features = np.array([[
        gender,
        entry_age,
        exit_age,
        duration,
        salary,
        prev_bal,
        contrib,
        inv,
        inf,
        real_return,
        prev_bal * duration,
        contrib  * duration,
        inv      * duration
    ]])
    return float(model.predict(features)[0])


def main():
    st.title("Pension Calculator")
    st.markdown("### Enter your details to get a projected benefit and sustainability analysis.")
    st.markdown("Click the **>>** on the left to open the panel and enter your details.")

    model = load_model()
    if model is None:
        st.error("Model could not be loaded. Please ensure 'Final_Pension_Model.pkl' is in the same folder as this script.")
        return

    with st.sidebar:
        st.header("Member Details")
        gender      = 1 if st.selectbox("Gender", ["Male", "Female"]) == "Male" else 0
        current_age = st.number_input("Current Age", min_value=18, max_value=80, value=35)
        retire_age  = st.number_input("Retirement Age", min_value=current_age+1, max_value=100, value=60)

        st.header("Financial Details")
        prev_bal = st.number_input("Accumulated Balance (USD)", value=5000.0)
        salary   = st.number_input("Current Annual Salary (USD)", value=15000.0)
        contrib  = st.number_input("Annual Contribution (USD)", value=1500.0)

        st.header("Economic Rates")
        inv_rate = st.number_input("Investment Rate (%)", value=8.0) / 100
        inf_rate = st.number_input("Inflation Rate (%)", value=5.0) / 100

        st.header("Advanced Features")
        enable_mc = st.checkbox("Enable Market Volatility (Monte Carlo)")
        vol_inv = st.number_input("Investment Volatility (%)", value=2.0) / 100 if enable_mc else 0
        vol_inf = st.number_input("Inflation Volatility (%)", value=2.0) / 100 if enable_mc else 0
        n_sims  = st.selectbox("No. of Simulations", [500, 1000]) if enable_mc else 0

        run_btn = st.button("Generate Report")

    if run_btn:
        duration = retire_age - current_age

        if enable_mc:
            sim_results = []
            for _ in range(n_sims):
                s_inv  = inv_rate + np.random.normal(0, vol_inv)
                s_inf  = inf_rate + np.random.normal(0, vol_inf)
                s_pred = predict_benefit(
                    model, gender,
                    current_age, retire_age, duration,
                    salary, prev_bal, contrib,
                    s_inv, s_inf
                )
                sim_results.append(max(0, s_pred))
            final_pred = np.mean(sim_results)
        else:
            final_pred = predict_benefit(
                model, gender,
                current_age, retire_age, duration,
                salary, prev_bal, contrib,
                inv_rate, inf_rate
            )

        final_pred = max(0, final_pred)

        st.markdown(f"""
        <div class="benefit-box">
            <p style="color:#1a3a5c; font-weight:bold; margin-bottom:5px;">ML-Projected Total Benefit at Retirement</p>
            <div class="benefit-value">${final_pred:,.2f}</div>
            {"<small>(Adjusted for investment and inflation volatility via Monte Carlo)</small>" if enable_mc else ""}
        </div>
        """, unsafe_allow_html=True)

        lump_sum     = final_pred * (1/3)
        annuity_pool = final_pred * (2/3)

        st.subheader("IPEC Regulatory Breakdown")
        c1, c2 = st.columns(2)
        with c1:
            st.info(f"**1/3 Cash Lump Sum:** ${lump_sum:,.2f}")
        with c2:
            st.success(f"**2/3 Residual for Annuity:** ${annuity_pool:,.2f}")

        st.divider()

        st.subheader("Benefit Sustainability")
        st.write("How many years will the 2/3 annuity pool last if you withdraw the following percentages of your current salary?")

        s1, s2, s3 = st.columns(3)
        rates = {"100% Salary": 1.0, "75% Salary": 0.75, "50% Salary": 0.50}

        for col, (label, rate) in zip([s1, s2, s3], rates.items()):
            annual_draw = salary * rate
            years       = annuity_pool / annual_draw if annual_draw > 0 else 0
            with col:
                if annual_draw > annuity_pool:
                    st.markdown(f"""
                    <div class="metric-box" style="border: 2px solid #ff4444;">
                        <div style="font-size:0.9rem;">Maintain {label}</div>
                        <div style="font-size:1.6rem; font-weight:bold; color:#ff4444;">⛔ Not Viable</div>
                        <div style="font-size:0.8rem;">(${annual_draw:,.0f}/year)</div>
                        <div style="font-size:0.75rem; color:#ff4444; margin-top:5px;">
                            Annual withdrawal (${annual_draw:,.0f}) exceeds your annuity pool (${annuity_pool:,.0f})
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="metric-box">
                        <div style="font-size:0.9rem;">Maintain {label}</div>
                        <div style="font-size:1.6rem; font-weight:bold;">{years:.1f} Years</div>
                        <div style="font-size:0.8rem;">(${annual_draw:,.0f}/year)</div>
                    </div>
                    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
