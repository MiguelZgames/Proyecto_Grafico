# Guía Técnica del Dashboard de Agentes

Este documento detalla el funcionamiento y la lógica de cada sección del dashboard para facilitar su interpretación y uso estratégico.

## 1. Perfil y Clasificación
**Funcionamiento:**
- **Clase:** Asignada automáticamente según el *Score Global* (ej. Diamond > 90, Platinum > 80).
- **Riesgo:** Un algoritmo secundario evalúa la volatilidad. Si el agente es impredecible, se marca como *Riesgo*; si es estable, *Seguro*.
**Valor:** Proporciona una evaluación inmediata del estatus del agente sin requerir análisis profundo.

## 2. Desempeño por Indicadores (KPIs)
**Funcionamiento:**
- **Barras Azules:** Representan el puntaje actual (0-10) en cada una de las 11 métricas.
- **Barras Grises:** Representan el potencial no realizado (la "brecha" hacia el 10 perfecto).
- **Tooltips:** Al pasar el cursor, se revela el **Peso Impacto**. Esto indica exactamente cuántos puntos del Score Global se ganarían al perfeccionar esa métrica.
**Valor:** Diagnóstico de precisión. Permite diferenciar si un bajo rendimiento se debe a problemas de volumen (financiero) o de calidad (retención/fidelidad).

## 3. Radar de Competitividad (Spider Chart)
**Funcionamiento:**
- Mapea las 11 métricas en un gráfico radial.
- **Comparación Multivariable:** Permite superponer perfiles de otros agentes. Las áreas de intersección muestran visualmente quién domina en qué aspecto.
- **Simetría:** Un polígono regular indica un agente equilibrado ("todoterreno"); picos agudos indican especialización.
**Valor:** Herramienta visual para *benchmarking*. Facilita entender las diferencias cualitativas entre categorías (ej. por qué un agente es Diamond y otro no).

## 4. Tendencia Histórica
**Funcionamiento:**
- Grafica la evolución mensual de **Depósitos, Retiros, GGR, NGR y Comisiones**.
- Permite visualizar la consistencia del rendimiento a lo largo del tiempo.
**Valor:** Contexto temporal. Identifica si un agente está en ascenso, en declive o si su rendimiento es estacional.

## 5. Análisis de Mejora (Próximo Nivel) -- Lógica Avanzada
**Funcionamiento: Similitud del Coseno**
Esta sección no utiliza promedios simples. Emplea álgebra vectorial para generar recomendaciones personalizadas.

1.  **Vectorización:** El perfil del agente se trata como un vector en un espacio de 11 dimensiones.
2.  **Objetivo (Centroide):** Se calcula el vector promedio de los agentes *exitosos* de la siguiente clase superior.
3.  **Cálculo Matemático:**
    $$ \text{Similitud} = \cos(\theta) = \frac{A \cdot B}{\|A\| \|B\|} $$
    Se mide el ángulo entre el vector del agente ($A$) y el vector objetivo ($B$).
4.  **Prescripción:** El sistema identifica qué métrica está "desviando" más el ángulo del vector y recomienda mejorar esa métrica específica para alinear al agente con el perfil de éxito.

**Valor:** Genera una hoja de ruta matemática y objetiva para el crecimiento, en lugar de consejos genéricos. Le dice al agente exactamente qué palanca mover para subir de nivel con el menor esfuerzo.
