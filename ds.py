import streamlit as st
import streamlit.components.v1 as components
import numpy as np
import pandas as pd
import sympy as sp
import math
import scipy.linalg as la
from scipy.interpolate import lagrange
from scipy.integrate import simpson, trapezoid
import plotly.graph_objects as go
import base64

# ==========================================
# PAGE CONFIG & STATE MANAGEMENT
# ==========================================
st.set_page_config(page_title="NMCP Pro Solver", layout="wide", page_icon="🧮")

if 'page' not in st.session_state: st.session_state.page = "Front Page"

def navigate(page):
    st.session_state.page = page

# ==========================================
# DYNAMIC CSS & CLEAN FLOATING FORMULAS
# ==========================================
def inject_dynamic_css():
    # Clean, semi-transparent math formulas
    svg_bg = """<svg width='1200' height='800' xmlns='http://www.w3.org/2000/svg'>
        <g fill='#4364F7' fill-opacity='0.08' font-family='Courier New, monospace' font-size='28' font-weight='bold'>
            <text x='50' y='150' transform='rotate(-10 50 150)'>x_{i+1} = x_i - f(x_i)/f'(x_i)</text>
            <text x='400' y='300' transform='rotate(15 400 300)'>[L][U]{x} = {b}</text>
            <text x='150' y='500'>y_{i+1} = y_i + (h/6)(k_1+2k_2+2k_3+k_4)</text>
            <text x='600' y='650' transform='rotate(-5 600 650)'>∫ f(x)dx ≈ (h/3)[y_0 + 4y_1 + y_2]</text>
            <text x='800' y='200' transform='rotate(10 800 200)'>P(x) = Σ y_i Π (x - x_j)/(x_i - x_j)</text>
            <text x='50' y='750' transform='rotate(-15 50 750)'>f(x) = Σ [f^(n)(a)/n!] * (x-a)^n</text>
        </g></svg>"""
    
    b64_svg = base64.b64encode(svg_bg.encode('utf-8')).decode('utf-8')

    st.markdown(f"""
    <style>
        @keyframes fadeIn {{ 0% {{ opacity: 0; transform: translateY(20px); }} 100% {{ opacity: 1; transform: translateY(0); }} }}
        @keyframes floatBg {{ 0% {{ background-position: 0px 0px; }} 100% {{ background-position: 1200px 800px; }} }}
        
        .block-container {{ animation: fadeIn 0.6s cubic-bezier(0.2, 0.8, 0.2, 1); padding-top: 2rem; }}
        
        /* Uses Streamlit's Native Colors to prevent muddy/bad color mixing */
        [data-testid="stAppViewContainer"] {{
            background-color: var(--background-color);
            background-image: url('data:image/svg+xml;base64,{b64_svg}');
            animation: floatBg 80s linear infinite;
        }}
        
        .main-title {{
            font-family: 'Segoe UI', sans-serif; font-size: 3.8rem; font-weight: 900;
            background: linear-gradient(135deg, #00c6ff, #0072ff);
            -webkit-background-clip: text; -webkit-text-fill-color: transparent;
            text-align: center; letter-spacing: -1px; text-shadow: 0px 5px 15px rgba(0, 114, 255, 0.2);
        }}
        .dept-title {{
            font-size: 1.4rem; font-weight: 700; color: #00c6ff; text-align: center;
            letter-spacing: 3px; text-transform: uppercase; margin-bottom: 30px;
        }}
        
        .glass-card {{
            padding: 30px; border-radius: 20px; border: 1px solid rgba(0, 198, 255, 0.3);
            background: var(--secondary-background-color);
            backdrop-filter: blur(15px); box-shadow: 0 15px 35px rgba(0,0,0,0.1);
            transition: transform 0.3s, box-shadow 0.3s;
        }}
        .glass-card:hover {{ transform: translateY(-5px); box-shadow: 0 20px 40px rgba(0, 198, 255, 0.2); }}
        
        .stButton>button {{
            font-weight: 700 !important; border-radius: 12px !important; border: 1px solid rgba(0, 198, 255, 0.5) !important;
            background: var(--secondary-background-color) !important; color: var(--text-color) !important;
            transition: all 0.3s ease !important; height: 60px; font-size: 1.1rem !important;
        }}
        .stButton>button:hover {{
            background: linear-gradient(135deg, #00c6ff, #0072ff) !important; color: white !important;
            box-shadow: 0 0 20px rgba(0, 198, 255, 0.4) !important; transform: scale(1.02);
        }}
        .math-block {{ background: var(--secondary-background-color); padding: 15px; border-left: 4px solid #00c6ff; border-radius: 8px; margin-bottom: 20px; overflow-x: auto; }}
    </style>
    """, unsafe_allow_html=True)

# ==========================================
# 3D WEBGL NETWORK ANIMATION
# ==========================================
def render_3d_network():
    html = """
    <!DOCTYPE html><html><head><style>body{margin:0;overflow:hidden;background:transparent;}</style>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script></head><body>
    <div id="canvas-container"></div><script>
        const scene = new THREE.Scene();
        const camera = new THREE.PerspectiveCamera(75, window.innerWidth/300, 0.1, 1000);
        const renderer = new THREE.WebGLRenderer({ alpha: true, antialias: true });
        renderer.setSize(window.innerWidth, 300);
        document.getElementById('canvas-container').appendChild(renderer.domElement);

        const particles = new THREE.BufferGeometry();
        const particleCount = 250;
        const posArray = new Float32Array(particleCount * 3);
        for(let i=0; i<particleCount*3; i++) posArray[i] = (Math.random() - 0.5) * 15;
        particles.setAttribute('position', new THREE.BufferAttribute(posArray, 3));
        
        const pMaterial = new THREE.PointsMaterial({ size: 0.1, color: 0x00c6ff });
        const pMesh = new THREE.Points(particles, pMaterial);
        scene.add(pMesh);
        
        const lineMaterial = new THREE.LineBasicMaterial({ color: 0x00c6ff, transparent: true, opacity: 0.15 });
        const lineMesh = new THREE.LineSegments(particles, lineMaterial);
        scene.add(lineMesh);

        camera.position.z = 6;
        let time = 0;
        function animate() {
            requestAnimationFrame(animate);
            pMesh.rotation.y += 0.002; pMesh.rotation.x += 0.001;
            lineMesh.rotation.y += 0.002; lineMesh.rotation.x += 0.001;
            renderer.render(scene, camera);
        }
        animate();
    </script></body></html>
    """
    components.html(html, height=300)

inject_dynamic_css()

# ==========================================
# PAGE 1: FRONT PAGE (ARCHITECT DASHBOARD)
# ==========================================
if st.session_state.page == "Front Page":
    render_3d_network()
    st.markdown('<div class="main-title" style="margin-top:-80px;">Numerical Methods & Computer Programming</div>', unsafe_allow_html=True)
    st.markdown('<div class="dept-title">Advanced Computational Engine</div>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("""
        <div class="glass-card">
            <h3 style="color: #00c6ff; text-align: center; margin-bottom: 20px; font-weight: 800;">System Architect</h3>
            <table style="width: 100%; font-size: 1.2rem; line-height: 2.0; color: var(--text-color);">
                <tr><td width="35%"><b>Name:</b></td><td>Varnabha patra</td></tr>
                <tr><td><b>Roll Number:</b></td><td>SEL24</td></tr>
                <tr><td><b>Institution:</b></td><td>Dr. D. Y. Patil Institute of Technology, Pimpri</td></tr>
                <tr><td><b>Department:</b></td><td>Department of Electrical Engineering </td></tr>
            </table>
        </div>
        """, unsafe_allow_html=True)
        st.button("Launch NMCP Modules 🚀", on_click=navigate, args=("Units",), use_container_width=True, type="primary")

# ==========================================
# PAGE 2: UNIT SELECTION DASHBOARD
# ==========================================
elif st.session_state.page == "Units":
    st.title("NMCP Syllabus Modules")
    st.caption("Select a mathematical unit to access its dedicated numerical solvers and simulators.")
    st.divider()

    c1, c2, c3 = st.columns([1, 2, 1])
    with c2:
        st.button("Unit I: Computational Errors & Taylor Series", on_click=navigate, args=("Unit I",), use_container_width=True)
        st.button("Unit II: Root Finding (Algebraic Equations)", on_click=navigate, args=("Unit II",), use_container_width=True)
        st.button("Unit III: Systems of Linear Equations", on_click=navigate, args=("Unit III",), use_container_width=True)
        st.button("Unit IV: Interpolation, Diff & Integration", on_click=navigate, args=("Unit IV",), use_container_width=True)
        st.button("Unit V: ODE Solvers & Least Squares Fit", on_click=navigate, args=("Unit V",), use_container_width=True)

    st.divider()
    st.button("← Terminate Session", on_click=navigate, args=("Front Page",))

# ==========================================
# UNIT I: ERRORS & TAYLOR SERIES
# ==========================================
elif st.session_state.page == "Unit I":
    st.title("Unit I: Computational Errors")
    st.caption("Analyzing Truncation Error using Taylor Series Expansions.")
    
    st.markdown('<div class="math-block">', unsafe_allow_html=True)
    st.latex(r"\text{Taylor Series: } f(x) = \sum_{n=0}^{N} \frac{f^{(n)}(x_0)}{n!} (x - x_0)^n")
    st.latex(r"\text{Absolute Error } (E_a) = | \text{True Value} - \text{Approx Value} |")
    st.latex(r"\text{Relative Error } (E_r) = \frac{E_a}{|\text{True Value}|} \quad \quad \text{Percentage Error } (\%E) = E_r \times 100\%")
    st.markdown('</div>', unsafe_allow_html=True)
    
    func_choice = st.selectbox("Select Function f(x)", ["sin(x)", "cos(x)", "exp(x)", "log(1+x)", "Custom Equation"])
    if func_choice == "Custom Equation":
        eq_str = st.text_input("Enter custom f(x) (e.g., x**2 + sin(x)):", "sin(x)")
    else:
        eq_str = func_choice
        
    c1, c2, c3 = st.columns(3)
    with c1: x0 = st.number_input("Expansion Point (x0)", value=0.0)
    with c2: x = st.number_input("Evaluation Point (x)", value=1.0)
    with c3: terms = st.number_input("Number of Terms (N)", value=8, max_value=20, min_value=1)
    
    if st.button("Compute Errors & Show Process", type="primary"):
        try:
            sym_x = sp.Symbol('x')
            sym_eq = sp.sympify(eq_str.replace("np.", ""))
            
            true_val = float(sym_eq.subs(sym_x, x))
            approx = 0.0
            errs = []
            steps = []
            
            for n in range(int(terms)):
                deriv = sp.diff(sym_eq, sym_x, n)
                d_val = float(deriv.subs(sym_x, x0))
                
                term_val = (d_val / math.factorial(n)) * ((x - x0)**n)
                approx += term_val
                
                abs_err = abs(true_val - approx)
                errs.append(abs_err)
                
                steps.append({
                    "n": n, 
                    "Derivative fⁿ(x)": str(deriv),
                    "fⁿ(x0)": round(d_val, 6), 
                    "Term Value": round(term_val, 6), 
                    "Approx (X_A)": round(approx, 6), 
                    "Abs Error (E_a)": round(abs_err, 6)
                })
            
            rel_err = errs[-1] / abs(true_val) if true_val != 0 else 0
            
            st.success(f"**True Value ($X_T$):** {true_val:.8f} | **Approx Value ($X_A$):** {approx:.8f}")
            st.info(f"**Final Absolute Error:** {errs[-1]:.8f} | **Relative Error:** {rel_err:.8f} | **Percentage Error:** {rel_err*100:.6f}%")
            
            with st.expander("🔍 View Step-by-Step Mathematical Process", expanded=True):
                st.dataframe(pd.DataFrame(steps), use_container_width=True)
                
            fig = go.Figure(go.Scatter(x=list(range(int(terms))), y=errs, mode='lines+markers', line=dict(color='#00c6ff', width=3)))
            fig.update_layout(title="Truncation Error Decay (Log Scale)", xaxis_title="Number of Terms (N)", yaxis_title="Absolute Error (Log)", yaxis_type="log", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e: 
            st.error(f"Mathematical Error: {e}. Ensure valid syntax.")
            
    st.divider()
    st.button("← Eject to Modules", on_click=navigate, args=("Units",))

# ==========================================
# UNIT II: ROOT FINDING
# ==========================================
elif st.session_state.page == "Unit II":
    st.title("Unit II: Algebraic & Transcendental Equations")
    method = st.selectbox("Algorithm", ["Bisection Method", "False Position (Regula Falsi)", "Newton-Raphson"])
    eq_str = st.text_input("f(x):", "x**3 - 4*x - 9")
    
    st.markdown('<div class="math-block">', unsafe_allow_html=True)
    if method == "Newton-Raphson":
        st.latex(r"x_{i+1} = x_i - \frac{f(x_i)}{f'(x_i)}")
    elif method == "Bisection Method":
        st.latex(r"c = \frac{a + b}{2} \quad \text{where} \quad f(a) \cdot f(b) < 0")
    else:
        st.latex(r"c = \frac{a \cdot f(b) - b \cdot f(a)}{f(b) - f(a)}")
    st.markdown('</div>', unsafe_allow_html=True)
    
    c1, c2 = st.columns(2)
    if method == "Newton-Raphson":
        with c1: x0 = st.number_input("Initial x0", value=2.0)
        with c2: tol = st.number_input("Tolerance", value=0.001, format="%f")
    else:
        with c1: a = st.number_input("Lower (a)", value=2.0)
        with c2: b = st.number_input("Upper (b)", value=3.0)
        tol = st.number_input("Tolerance", value=0.001, format="%f")

    if st.button("Solve & Show Process", type="primary"):
        try:
            x_sym = sp.Symbol('x')
            expr = sp.sympify(eq_str.replace("np.", ""))
            f_l = sp.lambdify(x_sym, expr, "numpy")
            df_l = sp.lambdify(x_sym, sp.diff(expr, x_sym), "numpy")
            
            steps = []
            if method == "Newton-Raphson":
                xi = x0
                for i in range(50):
                    fx, dfx = f_l(xi), df_l(xi)
                    steps.append({"Iter": i, "x_i": round(xi,6), "f(x_i)": round(fx,6), "f'(x_i)": round(dfx,6)})
                    if abs(fx) < tol: break
                    xi = xi - fx/dfx
                ans = xi
                plot_domain = [ans - 2, ans + 2]
            else:
                if f_l(a) * f_l(b) >= 0: st.error("f(a) and f(b) must have opposite signs."); st.stop()
                plot_domain = [a - 1, b + 1]
                for i in range(50):
                    c = (a+b)/2 if method == "Bisection Method" else (a*f_l(b) - b*f_l(a)) / (f_l(b) - f_l(a))
                    fc = f_l(c)
                    steps.append({"Iter": i, "a": round(a,4), "b": round(b,4), "c": round(c,6), "f(c)": round(fc,6)})
                    if abs(b-a) < tol or fc == 0: break
                    if f_l(a)*fc < 0: b = c
                    else: a = c
                ans = c
                
            st.success(f"**Root Found:** {ans:.6f}")
            with st.expander("🔍 View Step-by-Step Iteration Table", expanded=True):
                st.dataframe(pd.DataFrame(steps), use_container_width=True)
                
            fig = go.Figure()
            x_plot = np.linspace(plot_domain[0], plot_domain[1], 200)
            y_plot = f_l(x_plot)
            fig.add_trace(go.Scatter(x=x_plot, y=y_plot, mode='lines', name='f(x)', line=dict(color='#00c6ff')))
            fig.add_trace(go.Scatter(x=[ans], y=[f_l(ans)], mode='markers', name='Root', marker=dict(size=12, color='red', symbol='x')))
            fig.add_hline(y=0, line_dash="dash", line_color="gray")
            fig.update_layout(title=f"Interactive Function Plot for {eq_str}", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig, use_container_width=True)

        except Exception as e: st.error(f"Mathematical Error: Ensure function syntax is correct. Details: {e}")
        
    st.divider()
    st.button("← Eject to Modules", on_click=navigate, args=("Units",))

# ==========================================
# UNIT III: LINEAR SYSTEMS
# ==========================================
elif st.session_state.page == "Unit III":
    st.title("Unit III: Systems of Linear Equations")
    method = st.selectbox("Algorithm", ["Gauss Elimination", "Gauss Elimination (Partial Pivoting)", "Gauss-Jordan", "LU Decomposition", "Jacobi Iteration", "Gauss-Seidel Iteration"])
    n = st.number_input("System Dimension (N x N)", 2, 6, 3)
    
    st.markdown('<div class="math-block">', unsafe_allow_html=True)
    st.latex(r"[A] \{x\} = \{B\} \implies \begin{bmatrix} a_{11} & \dots & a_{1n} \\ \vdots & \ddots & \vdots \\ a_{n1} & \dots & a_{nn} \end{bmatrix} \begin{bmatrix} x_1 \\ \vdots \\ x_n \end{bmatrix} = \begin{bmatrix} b_1 \\ \vdots \\ b_n \end{bmatrix}")
    st.markdown('</div>', unsafe_allow_html=True)
    
    if 'mA' not in st.session_state or st.session_state.mat_n != n:
        st.session_state.mat_n = n
        st.session_state.mA = pd.DataFrame(np.eye(n)*4 - np.ones((n,n)), columns=[f"x{i+1}" for i in range(n)])
        st.session_state.mB = pd.DataFrame(np.ones((n, 1))*10, columns=["B"])

    c1, c2 = st.columns([3, 1])
    with c1: eA = st.data_editor(st.session_state.mA, use_container_width=True)
    with c2: eB = st.data_editor(st.session_state.mB, use_container_width=True)

    if st.button("Compute System & Show Steps", type="primary"):
        try:
            A, B = eA.values.astype(float), eB.values.astype(float).flatten()
            steps = []
            
            if "Gauss" in method and "Iteration" not in method:
                sol = np.linalg.solve(A, B)
                st.info("Direct Method applied via NumPy Linalg (LU + Partial Pivoting).")
            elif method == "LU Decomposition":
                lu, piv = la.lu_factor(A)
                sol = la.lu_solve((lu, piv), B)
                st.info("LU Decomposition applied via SciPy Linalg.")
            else:
                x = np.zeros(n)
                for iter in range(25): 
                    x_new = np.zeros(n)
                    step_dict = {"Iteration": iter+1}
                    for i in range(n):
                        if method == "Jacobi Iteration":
                            s = sum(A[i][j] * x[j] for j in range(n) if i != j)
                            x_new[i] = (B[i] - s) / A[i][i]
                        else: # Gauss-Seidel
                            s1 = sum(A[i][j] * x_new[j] for j in range(i))
                            s2 = sum(A[i][j] * x[j] for j in range(i + 1, n))
                            x_new[i] = (B[i] - s1 - s2) / A[i][i]
                        step_dict[f"x{i+1}"] = round(x_new[i], 5)
                    x = np.copy(x_new)
                    steps.append(step_dict)
                sol = x
                
            st.success("✅ System Computed Successfully!")
            cols = st.columns(len(sol))
            for i, val in enumerate(sol): cols[i].success(f"**x{i+1} = {val:.4f}**")
            
            if steps:
                with st.expander(f"🔍 View {method} Iteration Process", expanded=True):
                    st.dataframe(pd.DataFrame(steps), use_container_width=True)
        except Exception as e: st.error(f"Matrix Error: {e}. Check for singular matrices or non-diagonally dominant systems.")
        
    st.divider()
    st.button("← Eject to Modules", on_click=navigate, args=("Units",))

# ==========================================
# UNIT IV: INTERPOLATION & INTEGRATION
# ==========================================
elif st.session_state.page == "Unit IV":
    st.title("Unit IV: Interpolation, Diff & Integration")
    mode = st.radio("Operation", ["Interpolation/Diff", "Numerical Integration"], horizontal=True)
    
    if mode == "Interpolation/Diff":
        method = st.selectbox("Algorithm", ["Lagrange Interpolation", "Newton's Forward/Backward", "Numerical Differentiation"])
        
        st.markdown('<div class="math-block">', unsafe_allow_html=True)
        if method == "Lagrange Interpolation":
            st.latex(r"P(x) = \sum_{i=0}^n y_i \prod_{j \neq i} \frac{x - x_j}{x_i - x_j}")
        elif method == "Numerical Differentiation":
            st.latex(r"f'(x) \approx \frac{f(x+h) - f(x-h)}{2h}")
        else:
            st.latex(r"P(x) = y_0 + p \Delta y_0 + \frac{p(p-1)}{2!} \Delta^2 y_0 + \dots")
        st.markdown('</div>', unsafe_allow_html=True)
        
        if "Interpolation" in method or "Newton" in method:
            x_vals = st.text_input("X Data (comma separated)", "10, 20, 30, 40")
            y_vals = st.text_input("Y Data (comma separated)", "1.1, 2.0, 4.4, 7.9")
            x_t = st.number_input("Target X to Interpolate", value=25.0)
            
            if st.button("Interpolate & Graph", type="primary"):
                try:
                    x, y = np.array(list(map(float, x_vals.split(',')))), np.array(list(map(float, y_vals.split(','))))
                    poly = lagrange(x, y)
                    target_y = poly(x_t)
                    st.success(f"**Interpolated Value at X={x_t}:** {target_y:.6f}")
                    
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=x, y=y, mode='markers', name='Known Data', marker=dict(size=10, color='red')))
                    x_plot = np.linspace(min(x), max(x), 100)
                    fig.add_trace(go.Scatter(x=x_plot, y=poly(x_plot), mode='lines', name='Interpolated Polynomial', line=dict(color='#00c6ff')))
                    fig.add_trace(go.Scatter(x=[x_t], y=[target_y], mode='markers', name='Target Point', marker=dict(size=14, symbol='star', color='yellow')))
                    fig.update_layout(title="Polynomial Interpolation Curve", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e: st.error(f"Error: {e}")
        else:
            eq_str = st.text_input("f(x):", "sin(x)")
            x_eval = st.number_input("Evaluate at x =", value=0.785)
            h = st.number_input("Step size (h)", value=0.01, format="%f")
            if st.button("Calculate Derivative (Finite Diff)", type="primary"):
                try:
                    sym_x = sp.Symbol('x')
                    sym_eq = sp.sympify(eq_str.replace("np.", ""))
                    f_l = sp.lambdify(sym_x, sym_eq, "numpy")
                    diff = (f_l(x_eval + h) - f_l(x_eval - h)) / (2 * h)
                    st.success(f"**Derivative f'({x_eval}) ≈** {diff:.6f}")
                except Exception as e: st.error(f"Error: {e}")
                
    else:
        integ_method = st.selectbox("Rule", ["Trapezoidal Rule", "Simpson's 1/3 Rule", "Simpson's 3/8 Rule"])
        
        st.markdown('<div class="math-block">', unsafe_allow_html=True)
        if integ_method == "Simpson's 1/3 Rule":
            st.latex(r"\int_a^b f(x) dx \approx \frac{h}{3} \left[ y_0 + 4(y_1 + y_3 + ...) + 2(y_2 + y_4 + ...) + y_n \right]")
        elif integ_method == "Simpson's 3/8 Rule":
            st.latex(r"\int_a^b f(x) dx \approx \frac{3h}{8} \left[ y_0 + 3(y_1 + y_2 + y_4...) + 2(y_3 + y_6...) + y_n \right]")
        else:
            st.latex(r"\int_a^b f(x) dx \approx \frac{h}{2} \left[ (y_0 + y_n) + 2(y_1 + y_2 + ... + y_{n-1}) \right]")
        st.markdown('</div>', unsafe_allow_html=True)
        
        eq = st.text_input("Integrand f(x):", "sin(x)")
        c1, c2, c3 = st.columns(3)
        with c1: a = st.number_input("Lower Bound", value=0.0)
        with c2: b = st.number_input("Upper Bound", value=3.14159)
        with c3: n = st.number_input("Intervals (n)", value=12, step=1)
        
        if st.button("Integrate & Visualize Area", type="primary"):
            try:
                sym_x = sp.Symbol('x')
                sym_eq = sp.sympify(eq.replace("np.", ""))
                f_l = sp.lambdify(sym_x, sym_eq, "numpy")
                
                n = int(n)
                x = np.linspace(a, b, n+1)
                y = f_l(x)
                h = (b-a)/n
                
                if integ_method == "Trapezoidal Rule":
                    ans = trapezoid(y, x)
                elif integ_method == "Simpson's 1/3 Rule":
                    if n % 2 != 0: st.warning("Note: Simpson's 1/3 rule mathematically requires an even number of intervals.")
                    ans = simpson(y, x=x)
                else:
                    if n % 3 != 0: st.warning("Note: Simpson's 3/8 rule mathematically requires 'n' to be a multiple of 3.")
                    ans = (3*h/8) * (y[0] + 3*np.sum(y[1:-1:3]) + 3*np.sum(y[2:-1:3]) + 2*np.sum(y[3:-1:3]) + y[-1])
                
                st.success(f"**Calculated Integral Area:** {ans:.6f}")
                
                fig = go.Figure()
                xs = np.linspace(a, b, 200)
                ys = f_l(xs)
                fig.add_trace(go.Scatter(x=xs, y=ys, mode='lines', line=dict(color='#00c6ff', width=3), name='f(x)'))
                fig.add_trace(go.Scatter(x=x, y=y, fill='tozeroy', mode='none', fillcolor='rgba(0, 198, 255, 0.3)', name='Calculated Area'))
                fig.update_layout(title="Numerical Area Integration Shading", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e: st.error(f"Error: {e}")
            
    st.divider()
    st.button("← Eject to Modules", on_click=navigate, args=("Units",))

# ==========================================
# UNIT V: ODEs & CURVE FITTING
# ==========================================
elif st.session_state.page == "Unit V":
    st.title("Unit V: ODEs & Least Squares Curve Fitting")
    mode = st.radio("Operation", ["Solve ODE", "Least Squares Curve Fitting"], horizontal=True)
    
    st.markdown('<div class="math-block">', unsafe_allow_html=True)
    if mode == "Solve ODE":
        st.latex(r"y_{i+1} = y_i + \frac{h}{6}(k_1 + 2k_2 + 2k_3 + k_4) \quad \text{(RK4)}")
    else:
        st.latex(r"S_r = \sum_{i=1}^n (y_i - f(x_i))^2")
    st.markdown('</div>', unsafe_allow_html=True)
    
    if mode == "Solve ODE":
        ode_method = st.selectbox("Algorithm", ["Euler's Method", "Modified Euler Method", "Runge-Kutta 2nd Order", "Runge-Kutta 4th Order (RK4)"])
        eq_str = st.text_input("dy/dt = f(t,y):", "10 - 2*y")
        c1, c2, c3 = st.columns(3)
        with c1: t0 = st.number_input("Initial t0", value=0.0)
        with c2: y0 = st.number_input("Initial y0", value=0.0)
        with c3: h = st.number_input("Step size (h)", value=0.1)
        steps_cnt = st.number_input("Number of Steps to Simulate", value=20)
        
        if st.button("Solve ODE & Show Steps", type="primary"):
            try:
                t_s, y_s = sp.symbols('t y')
                f = sp.lambdify((t_s, y_s), sp.sympify(eq_str.replace("np.", "")), "numpy")
                t, y, steps, tl, yl = t0, y0, [], [t0], [y0]
                
                for i in range(int(steps_cnt)): 
                    if ode_method == "Euler's Method":
                        dy = f(t, y)
                        steps.append({"Iter": i, "t": round(t,3), "y": round(y,4), "f(t,y)": round(dy,4)})
                        y = y + h * dy
                    elif ode_method in ["Modified Euler Method", "Runge-Kutta 2nd Order"]:
                        k1 = h * f(t, y)
                        k2 = h * f(t + h, y + k1)
                        steps.append({"Iter": i, "t": round(t,3), "y": round(y,4), "k1": round(k1,4), "k2": round(k2,4)})
                        y = y + 0.5 * (k1 + k2)
                    else: # RK4
                        k1 = h * f(t, y)
                        k2 = h * f(t + h/2, y + k1/2)
                        k3 = h * f(t + h/2, y + k2/2)
                        k4 = h * f(t + h, y + k3)
                        steps.append({"Iter": i, "t": round(t,3), "y": round(y,4), "k1": round(k1,4), "k2": round(k2,4), "k3": round(k3,4), "k4": round(k4,4)})
                        y = y + (1/6)*(k1 + 2*k2 + 2*k3 + k4)
                        
                    t += h
                    tl.append(t); yl.append(y)
                
                st.success(f"**Final State computed at t={t:.2f}:** {y:.6f}")
                with st.expander(f"🔍 View {ode_method} Iteration Variables", expanded=True):
                    st.dataframe(pd.DataFrame(steps), use_container_width=True)
                    
                fig = go.Figure(go.Scatter(x=tl, y=yl, mode='lines+markers', line=dict(color='#00c6ff', width=3)))
                fig.update_layout(title=f"Transient System Response ({ode_method})", xaxis_title="Time (t)", yaxis_title="State (y)", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e: st.error(f"Error: {e}. Check function syntax (use variables t and y).")
            
    else:
        fit_method = st.selectbox("Polynomial Degree", ["Linear (1st Order)", "Quadratic (2nd Order)"])
        x_d = st.text_input("Independent X Data", "10, 20, 30, 40, 50")
        y_d = st.text_input("Dependent Y Data", "15, 35, 41, 60, 85")
        
        if st.button("Generate Regression Model", type="primary"):
            try:
                x, y = np.array(list(map(float, x_d.split(',')))), np.array(list(map(float, y_d.split(','))))
                deg = 1 if "Linear" in fit_method else 2
                coeffs = np.polyfit(x, y, deg)
                poly = np.poly1d(coeffs)
                st.success(f"**Fitted Polynomial Equation:**\n y = {poly}")
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=x, y=y, mode='markers', name='Experimental Data', marker=dict(size=10, color='red')))
                xp = np.linspace(min(x), max(x), 50)
                fig.add_trace(go.Scatter(x=xp, y=poly(xp), mode='lines', name='Least Squares Fit', line=dict(color='#00c6ff', width=3)))
                fig.update_layout(title="Least Squares Curve Fitting", xaxis_title="X", yaxis_title="Y", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e: st.error(f"Data Error: {e}. Ensure identical lengths of X and Y comma-separated arrays.")

    st.divider()
    st.button("← Eject to Modules", on_click=navigate, args=("Units",))
