library(shiny)
library(tidyverse)
library(caret)
library(randomForest)


# 1. Load Data & Model


heartData <- read.csv("synthetic_heart_disease_dataset.csv")

heartData <- heartData %>%
  mutate(
    Gender            = as.factor(Gender),
    Smoking           = as.factor(Smoking),
    Alcohol_Intake    = as.factor(Alcohol_Intake),
    Physical_Activity = as.factor(Physical_Activity),
    Diet              = as.factor(Diet),
    Stress_Level      = as.factor(Stress_Level),
    Heart_Disease     = as.factor(Heart_Disease)
  )

rf_model <- readRDS("heart_rf_model.rds")

set.seed(123)
trainIndex <- createDataPartition(heartData$Heart_Disease,
                                  p = 0.7, list = FALSE)
trainData <- heartData[trainIndex, ]
testData  <- heartData[-trainIndex, ]
test_pred <- predict(rf_model, newdata = testData)
test_accuracy <- mean(test_pred == testData$Heart_Disease)

# averages for defaults
avg_weight    <- round(mean(heartData$Weight), 1)
avg_height    <- round(mean(heartData$Height), 1)
avg_bmi       <- round(mean(heartData$BMI), 1)
avg_sys       <- round(mean(heartData$Systolic_BP), 0)
avg_dia       <- round(mean(heartData$Diastolic_BP), 0)
avg_hr        <- round(mean(heartData$Heart_Rate), 0)
avg_sugar     <- round(mean(heartData$Blood_Sugar_Fasting), 0)
avg_chol      <- round(mean(heartData$Cholesterol_Total), 0)


# 2. UI


ui <- fluidPage(
  tags$head(
    tags$title("Heart Disease Prediction"),
    tags$style(HTML("
      body {
        margin: 0;
        padding: 0;
        background-color: #000000;
        background-image: url('heart_bg.jpg');
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
        color: white;
        font-family: 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
      }

      .top-banner {
        width: 100%;
        text-align: center;
        padding: 6px 10px;
        background-color: rgba(0,0,0,0.7);
        font-size: 12px;
        letter-spacing: 0.3px;
      }

      /* container for the home page content */
      .app-background {
        min-height: 100vh;
        display: flex;
        align-items: center;
        justify-content: center;
        padding: 40px 15px;
      }

      /* Glass card for inputs */
      .card-box {
        max-width: 1150px;
        width: 100%;
        background-color: rgba(0, 0, 0, 0.78);
        padding: 30px 40px 35px 40px;
        border-radius: 24px;
        box-shadow: 0 0 30px rgba(0,0,0,0.9);
      }

      .card-title {
        text-align: center;
        font-size: 32px;
        font-weight: 700;
        margin-bottom: 20px;
        color: #ffffff;
      }

      .shiny-input-container {
        margin-bottom: 12px;
      }

      label {
        font-size: 13px;
        font-weight: 500;
      }

      input[type='number'],
      .selectize-input,
      .form-control {
        background-color: rgba(255,255,255,0.96) !important;
        color: #000000 !important;
        border-radius: 8px !important;
        border: none !important;
        box-shadow: 0 0 0 1px rgba(0,0,0,0.15);
      }

      .selectize-input {
        width: 100% !important;
        box-sizing: border-box;
      }

      .btn-primary {
        font-weight: 600;
        height: 44px;
        border-radius: 999px;
      }

      /* RESULT PAGE STYLING - full black overlay */
      .result-page {
        position: fixed;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background-color: #000000;
        display: flex;
        align-items: center;
        justify-content: center;
        padding: 20px;
        z-index: 9999;
      }

      .result-wrapper {
        text-align: center;
      }

      .result-title-out {
        font-size: 34px;
        font-weight: 700;
        color: #ffffff;
        margin-bottom: 20px;
      }

      .result-card {
        max-width: 900px;
        width: 90%;
        background-color: #333333;
        padding: 40px 50px;
        border-radius: 24px;
        box-shadow: 0 0 40px rgba(0,0,0,0.95);
        text-align: center;
        margin: 0 auto;
      }

      #resultText {
        font-size: 22px;
        font-weight: 600;
        margin-bottom: 18px;
      }

      #adviceText {
        font-size: 17px;
        line-height: 1.7;
        color: #f0f0f0;
      }

      .back-button {
        margin-top: 28px;
        width: 190px;
        height: 42px;
        font-weight: 600;
        border-radius: 999px;
      }
    "))
  ),
  
  div(
    class = "top-banner",
    "Predictions of this model are not always accurate. Please consult a doctor."
  ),
  
  tabsetPanel(
    id = "main_tabs",
    type = "hidden",
    
    # ------------ HOME / INPUT PAGE ------------
    tabPanel(
      "Input",
      div(
        class = "app-background",
        div(
          class = "card-box",
          div(class = "card-title", "Heart Disease Prediction"),
          
          fluidRow(
            # ===== Column 1 – Core vitals =====
            column(
              width = 4,
              numericInput("Age", "Age", value = 50, min = 18, max = 100),
              selectInput(
                "Gender",
                "Sex",
                choices = levels(heartData$Gender)
              ),
              numericInput(
                "Systolic_BP",
                "Resting Systolic Blood Pressure:",
                value = avg_sys, min = 80, max = 220
              ),
              numericInput(
                "Diastolic_BP",
                "Resting Diastolic Blood Pressure:",
                value = avg_dia, min = 40, max = 140
              ),
              numericInput(
                "Heart_Rate",
                "Resting Heart Rate:",
                value = avg_hr, min = 40, max = 200
              ),
              selectInput(
                "Hypertension",
                "Hypertension (High BP) (0 = No, 1 = Yes):",
                choices = c(0, 1)
              )
            ),
            
            # ===== Column 2 – Lab values & body metrics =====
            column(
              width = 4,
              numericInput(
                "Blood_Sugar_Fasting",
                "Fasting Blood Sugar:",
                value = avg_sugar, min = 60, max = 300
              ),
              numericInput(
                "Cholesterol_Total",
                "Serum Cholesterol:",
                value = avg_chol, min = 100, max = 400
              ),
              numericInput(
                "BMI",
                "Body Mass Index (BMI):",
                value = avg_bmi, min = 10, max = 60
              ),
              numericInput(
                "Weight",
                "Weight (kg):",
                value = avg_weight, min = 30, max = 200
              ),
              numericInput(
                "Height",
                "Height (cm):",
                value = avg_height, min = 120, max = 220
              ),
              selectInput(
                "Diabetes",
                "Diabetes (0 = No, 1 = Yes):",
                choices = c(0, 1)
              ),
              selectInput(
                "Hyperlipidemia",
                "Hyperlipidemia (0 = No, 1 = Yes):",
                choices = c(0, 1)
              )
            ),
            
            # ===== Column 3 – Lifestyle & history =====
            column(
              width = 4,
              selectInput(
                "Smoking",
                "Smoking:",
                choices = levels(heartData$Smoking)
              ),
              selectInput(
                "Alcohol_Intake",
                "Alcohol Intake:",
                choices = levels(heartData$Alcohol_Intake)
              ),
              selectInput(
                "Physical_Activity",
                "Physical Activity:",
                choices = levels(heartData$Physical_Activity)
              ),
              selectInput(
                "Diet",
                "Diet Quality:",
                choices = levels(heartData$Diet)
              ),
              selectInput(
                "Stress_Level",
                "Stress Level:",
                choices = levels(heartData$Stress_Level)
              ),
              selectInput(
                "Family_History",
                "Family History of Heart Disease (0 = No, 1 = Yes):",
                choices = c(0, 1)
              ),
              selectInput(
                "Previous_Heart_Attack",
                "Previous Heart Attack (0 = No, 1 = Yes):",
                choices = c(0, 1)
              )
            )
          ),
          
          br(),
          div(
            style = "text-align:center;",
            actionButton("predictBtn", "Predict", class = "btn btn-primary btn-lg")
          )
        )
      )
    ),
    
    # ------------ RESULT PAGE ------------
    tabPanel(
      "Result",
      div(
        class = "result-page",
        div(
          class = "result-wrapper",
          div(class = "result-title-out", "Heart Disease Prediction Result"),
          div(
            class = "result-card",
            div(id = "resultText", textOutput("resultText")),
            br(),
            div(id = "adviceText", htmlOutput("adviceText")),
            br(),
            div(
              style = "text-align:center;",
              actionButton(
                "backBtn",
                "Back to Home",
                class = "btn btn-secondary back-button"
              )
            )
          )
        )
      )
    )
  )
)


# 3. Server


server <- function(input, output, session) {
  
  rv <- reactiveValues(pred = NULL, prob = NULL)
  
  observeEvent(input$predictBtn, {
    
    new_patient <- tibble(
      Age = input$Age,
      Gender = factor(input$Gender, levels = levels(heartData$Gender)),
      Weight = input$Weight,
      Height = input$Height,
      BMI = input$BMI,
      Smoking = factor(input$Smoking, levels = levels(heartData$Smoking)),
      Alcohol_Intake = factor(input$Alcohol_Intake,
                              levels = levels(heartData$Alcohol_Intake)),
      Physical_Activity = factor(
        input$Physical_Activity,
        levels = levels(heartData$Physical_Activity)
      ),
      Diet = factor(input$Diet, levels = levels(heartData$Diet)),
      Stress_Level = factor(
        input$Stress_Level,
        levels = levels(heartData$Stress_Level)
      ),
      Hypertension = as.numeric(input$Hypertension),
      Diabetes = as.numeric(input$Diabetes),
      Hyperlipidemia = as.numeric(input$Hyperlipidemia),
      Family_History = as.numeric(input$Family_History),
      Previous_Heart_Attack = as.numeric(input$Previous_Heart_Attack),
      Systolic_BP = input$Systolic_BP,
      Diastolic_BP = input$Diastolic_BP,
      Heart_Rate = input$Heart_Rate,
      Blood_Sugar_Fasting = input$Blood_Sugar_Fasting,
      Cholesterol_Total = input$Cholesterol_Total
    )
    
    pred_class <- predict(rf_model, newdata = new_patient)
    prob_matrix <- predict(rf_model, newdata = new_patient, type = "prob")
    prob_1 <- prob_matrix[1, "1"]
    
    rv$pred <- as.character(pred_class)
    rv$prob <- prob_1
    
    updateTabsetPanel(session, "main_tabs", selected = "Result")
  })
  
  output$resultText <- renderText({
    req(rv$pred)
    
    if (rv$pred == "1") {
      "Unfortunately, heart disease risk has been detected by the model."
    } else {
      "Great news! The model does not detect heart disease risk."
    }
  })
  
  output$adviceText <- renderUI({
    req(rv$pred, rv$prob)
    
    prob_msg <- paste0(
      "Estimated probability of heart disease: ",
      round(rv$prob * 100, 2), "%."
    )
    
    if (rv$pred == "1") {
      HTML(paste0(
        "Heart disease has been detected by this model. ",
        "It's important to consult with a healthcare professional and consider ",
        "lifestyle changes such as a heart-healthy diet, regular exercise, ",
        "and stress management.",
        "<br><br>",
        prob_msg
      ))
    } else {
      HTML(paste0(
        "No heart disease has been detected by this model. ",
        "Keep maintaining a healthy lifestyle with regular exercise, a balanced diet, ",
        "and routine health check-ups.",
        "<br><br>",
        prob_msg
      ))
    }
  })
  
  observeEvent(input$backBtn, {
    updateTabsetPanel(session, "main_tabs", selected = "Input")
  })
}


# 4. Run App


shinyApp(ui = ui, server = server)
