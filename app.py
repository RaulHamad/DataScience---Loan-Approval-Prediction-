import pickle
import pandas as pd
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.compose import ColumnTransformer
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from shiny import reactive,req
from shiny.express import input,ui,render
import faicons as fa
from faicons import icon_svg as icon
from datetime import datetime

 

# Cargar one hot encoder
with open("one_hot_encoder.pkl", "rb") as f:
    one_hot_encoder_prestamo = pickle.load(f)

# Cargar scaler
with open("scaler.pkl", "rb") as f:
    scaler_prestamo = pickle.load(f)

#Cargar modelo
with open("model_aprendizaje.sav", "rb") as f:
    random_forest_prestamo = pickle.load(f)


ui.page_opts(fillable=True)
class_="bg-blue-500 text-white p-4 rounded"
ui.tags.h2(
    "Banco XXX - Préstamos",
    ui.tags.br(),
    "Introduzca los datos para comprobar la aprobación del préstamo",
    class_="text-xl sm:text-2xl md:text-3xl lg:text-4xl font-bold text-center text-white bg-dark",
    
)



with ui.card(class_="bg-body-secondary"):
    with ui.layout_column_wrap():

        with ui.value_box(showcase=icon("people-group")):
            ui.tags.h4(
                "Número Dependientes",
                class_="text-xl sm:text-2xl md:text-3xl lg:text-4xl font-bold text-center mb-4",
            )
            
            ui.input_slider("n_dependientes", "", 0, 10, 0)
            ui.tags.br()

        with ui.value_box(showcase=icon("graduation-cap")):
            ui.tags.h4(
                "Educación",
                class_="text-xl sm:text-2xl md:text-3xl lg:text-4xl font-bold text-center mb-4",
            )
            ui.input_selectize(
                "educacion",
                "",
                {"Graduate": "Graduate", "Not Graduate": "Not Graduate"},
            )
            ui.tags.br()
            ui.tags.br()
        with ui.value_box(showcase=icon("briefcase")):
            ui.tags.h4(
                "Trabajador Autonomo",
                class_="text-xl sm:text-2xl md:text-3xl lg:text-4xl font-bold text-center mb-4",
            )
            ui.input_radio_buttons(
                "trabajador_autonomo",
                "",
                {"Yes": "Yes", "No": "No"},
            )
            ui.tags.br()
        with ui.value_box(showcase=icon("wallet")):
            ui.tags.h4(
                "Ingresos Anuales",
                class_="text-xl sm:text-2xl md:text-3xl lg:text-4xl font-bold text-center mb-4",
            )
            ui.input_numeric("Ingresos_Anuales", "", value=0)
            ui.tags.br()
            ui.tags.br()
        with ui.value_box(showcase=icon("piggy-bank")):
            ui.tags.h4(
                "Valor Prestamo",
                class_="text-xl sm:text-2xl md:text-3xl lg:text-4xl font-bold text-center mb-4",
            )
            ui.input_numeric("Valor_Prestamo", "", value=0)
            ui.tags.br()
            
        with ui.value_box(showcase=icon("clock")):
            ui.tags.h4(
                "Plazo Prestamo",
                class_="text-xl sm:text-2xl md:text-3xl lg:text-4xl font-bold text-center mb-4",
            )
            ui.input_slider("Plazo_Prestamo", "", 0, 100, 0)
            ui.tags.br()
        with ui.value_box(showcase=icon("credit-card")):
            ui.tags.h4(
                "Pontuación Crédito",
                class_="text-xl sm:text-2xl md:text-3xl lg:text-4xl font-bold text-center mb-4",
            )
            ui.input_numeric("Pontuacion_Credito", "", value=0)
            ui.tags.br()
            ui.tags.br()
        with ui.value_box(showcase=icon("dollar-sign")):
            ui.tags.h4(
                "Activos Residenciales",
                class_="text-xl sm:text-2xl md:text-3xl lg:text-4xl font-bold text-center mb-4",
            )
            ui.input_numeric("Activos_Residenciales", "", value=0)
            ui.tags.br()
            ui.tags.br()
        with ui.value_box(showcase=icon("store")):
            ui.tags.h4(
                "Activos Comerciales",
                class_="text-xl sm:text-2xl md:text-3xl lg:text-4xl font-bold text-center mb-4",
            )
            ui.input_numeric("Activos_Comerciales", "", value=0)
            ui.tags.br()
            ui.tags.br()
        with ui.value_box(showcase=icon("gem")):
            ui.tags.h4(
                "Activos Lujo",
                class_="text-xl sm:text-2xl md:text-3xl lg:text-4xl font-bold text-center mb-4",
            )
            ui.input_numeric("Activos_Lujo", "", value=0)
            ui.tags.br()
        with ui.value_box(showcase=icon("building-columns")):
            ui.tags.h4(
                "Activo Bancário",
                class_="text-xl sm:text-2xl md:text-3xl lg:text-4xl font-bold text-center mb-4",
            )
            ui.input_numeric("Activo_Bancário", "", value=0)
            ui.tags.br()
    ui.input_action_button("action_button", "Previsión de Prestamo",class_="btn-primary")
    





@reactive.event(input.action_button)
def result():
    req(input.action_button())  # garante que o botão foi clicado
    # (mesma lógica de previsão aqui)
    columns = ['N Dependientes','Educación','Trabajador Autonomo','Ingresos Anuales','Valor Préstamo',
        'Plazo Préstamo','Pontuación Crédito','Valor Activos Residenciales','Valor Activos Comerciales',
        'Valor Activos Lujo','Valor Activo Bancário']
    # Pegar os dados dos inputs
    datos = pd.DataFrame([{
        "N Dependientes": input.n_dependientes(),
        "Educación": input.educacion(),
        "Trabajador Autonomo": input.trabajador_autonomo(),
        "Ingresos Anuales": input.Ingresos_Anuales(),
        "Valor Préstamo": input.Valor_Prestamo(),
        "Plazo Préstamo": input.Plazo_Prestamo(),
        "Pontuación Crédito": input.Pontuacion_Credito(),
        "Valor Activos Residenciales": input.Activos_Residenciales(),
        "Valor Activos Comerciales": input.Activos_Comerciales(),
        "Valor Activos Lujo": input.Activos_Lujo(),
        "Valor Activo Bancário": input.Activo_Bancário()
        
    }],columns=columns)
    
    #numpy
    datos = datos.values
                        
        
    #One hotencoder
    
    datos = one_hot_encoder_prestamo.transform(datos)
    
    
    #Scaler
    
    datos = scaler_prestamo.transform(datos)

    # Previsão
    
    prev = random_forest_prestamo.predict(datos)
    
    status = "✅ Aprobado ✅" if prev[0] == "Approved" else "❌ No Aprobado ❌"
    return f"Resultado para el prestamo: {status}"

    
@render.text
def previsao():
    input.action_button()
    return result()

