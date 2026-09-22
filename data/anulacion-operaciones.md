# PRD-INV-001 — Anulación de Operaciones (v2.0)

Documentos relacionados: PRD-TAB-001 Tabla de Operaciones, PRD-TAB-001
Congelamiento de Fondos.

## 1. Contexto y objetivo

El presente documento describe las reglas de negocio para la anulación y
eliminación de las operaciones dentro de la plataforma Alphinance. Abarca
los procesos de anulación individual y masiva, la revaluación de saldos y
tenencias, la consistencia de identificadores entre estados, y los permisos
operativos por rol. Establece la lógica de ventana de eliminación de
acuerdo al período habilitado en parametría, el impacto en tiempo real para
todas las operaciones y anulaciones, e incorpora un modelo formal de saldos
negativos.

## 2. Alcance funcional

### 2.1 Anulación de operaciones

El sistema permite anular operaciones registradas bajo las siguientes
condiciones:

- **Sin restricción de orden**: se puede anular cualquier operación sin
  importar su fecha de concertación, su posición en la secuencia o la
  existencia de operaciones posteriores en la misma cuenta.
- **Sin bloqueo por dependencias**: la existencia de operaciones con fecha
  posterior no impide la anulación. El sistema absorbe las inconsistencias
  resultantes mediante saldos negativos.
- **Sin intervención de soporte**: toda anulación posible dentro del
  período habilitado se resuelve por interfaz, sin necesidad de escalar
  tickets a soporte.

Ejemplo: si existen operaciones del ID 1 al 15, el usuario puede anular
directamente la n.° 13 sin necesidad de anular primero las operaciones 14 y
15.

### 2.2 Período habilitado por parametría

La única restricción temporal vigente es el período habilitado en la
parametría del módulo. El usuario podrá cargar o anular operaciones
únicamente dentro de dicho período (definido por fecha de inicio y fecha de
cierre).

### 2.3 Anulaciones masivas

El usuario podrá realizar anulaciones masivas mediante checkbox en la tabla
de operaciones, debiendo justificar la eliminación a través de un
comentario (motivo, vía modal) que queda persistido en el detalle de cada
operación una vez ejecutada la anulación. El sistema procesa el lote
completo y actualiza los saldos en tiempo real.

### 2.4 Recarga post-anulación

Una vez anulada una operación, el usuario puede volver a registrarla con
los ajustes necesarios. La nueva operación recibe un nuevo ID según la
secuencia vigente.

### 2.5 Revaluación en tiempo real

Toda operación registrada, modificada o anulada impacta en tiempo real
sobre: tenencias (cantidades y valores de posición), rentabilidad
(resultados y métricas de rendimiento), inventario (composición de la
cartera) y dashboards (gráficos de Beta, Sharpe ratio y Performance).

### 2.6 Lógica de IDs entre pestañas

El ID utilizado en la pestaña "Pendiente Control" es el mismo que el de
"Pendiente Bancaria", permitiendo el seguimiento a lo largo del flujo de
aprobación. Ambas pestañas mantienen numeración ascendente, y pueden
existir saltos en la secuencia por aprobaciones o rechazos en distintos
puntos del flujo. El ID de una operación en estado "Registrada" es
diferente al de los estados previos: la tabla de registradas inicia una
nueva secuencia ascendente independiente.

### 2.7 Permisos por rol — rechazo de operaciones

Los usuarios con rol junior o senior podrán rechazar sus propias
operaciones en estado "Pendiente Control" o "Pendiente Bancaria", siempre
que la operación haya sido ingresada por el mismo usuario y no haya
impactado aún en la tenencia. Esta regla permite corregir errores
operativos sin requerir intervención de un rol autorizante, liberando
saldos "congelados" de acuerdo al PRD de Congelamiento de Fondos.

### 2.8 Modelo de saldos negativos

El sistema acepta y registra saldos negativos tanto en efectivo como en
tenencias de activos, como resultado de anulaciones.

Comportamiento con saldo negativo:

- **Bloqueado**: operaciones de egreso de fondos o ventas de activos sobre
  la posición negativa.
- **Permitido**: operaciones de ingreso de fondos o compras de activos que
  reviertan o compensen el saldo negativo.

El estado negativo no restringe otras cuentas ni otros instrumentos, solo
las cuentas o instrumentos afectados. Toda situación que genere saldo
negativo queda registrada en el log de auditoría.

## 3. Alcance no funcional

- Actualización de saldos y tenencias en tiempo real.
- Soporte de valores negativos en todos los campos numéricos: saldo de
  efectivo, tenencia de activos y rentabilidad.
- Log de auditoría completo por cada anulación: usuario, timestamp, IP,
  motivo, saldos pre/post operación.
- Trazabilidad de IDs entre pestañas conservada post-anulación.
- Seguridad: todas las acciones de anulación respetan los permisos
  definidos por rol en el sistema.

## 4. Dependencias y riesgos

### 4.1 Dependencias

- Módulo de Parametría: debe exponer la fecha de inicio y cierre del
  período activo para ser consumida por el módulo de operaciones en tiempo
  real.
- Motor de Tenencias y Saldos: debe soportar valores negativos y
  actualización en tiempo real.
- Motor de Rentabilidad: debe soportar tenencias y efectivo negativo,
  además de reversar las operaciones afectadas por la anulación.
- PRD-TAB-001 Congelamiento de Fondos: el saldo comprometido se calcula
  sobre el saldo disponible. Si el saldo total es negativo, el disponible
  es cero y no se pueden comprometer fondos para nuevas operaciones de
  egreso.
- Log de auditoría: debe registrar cada anulación con usuario, timestamp,
  motivo y valores pre/post.

### 4.2 Riesgos

| Riesgo | Mitigación | Prioridad |
| --- | --- | --- |
| Saldo negativo persistente sin corrección | Alerta configurable en dashboard para administradores sobre cuentas con saldo negativo prolongado (deseable) | Baja |

## 5. Reglas de negocio (Dado / Cuando / Entonces)

**5.1 Período habilitado por parametría** — Dado que un usuario intenta
cargar o anular una operación, cuando accede al formulario de carga o a la
acción de anulación, entonces el sistema valida que la fecha de la
operación se encuentre dentro del período habilitado en la parametría. Si
la fecha está dentro del período, la operación se habilita sin
restricciones adicionales de fecha. Si está fuera del período, se bloquea y
se muestra: "El período seleccionado se encuentra cerrado. Contacte al
administrador para habilitar el período correspondiente."

**5.2 Revaluación en tiempo real** — Dado que se registra o anula una
operación dentro del período habilitado, cuando la operación es confirmada
(aprobada o registrada directamente), entonces el sistema recalcula y
actualiza en tiempo real tenencias, rentabilidad, inventario y dashboards.
El impacto es inmediato y visible para el usuario.

**5.3 Anulación sin restricciones por dependencias** — Dado que existe una
operación registrada dentro del período habilitado, con o sin operaciones
posteriores en la misma cuenta o instrumento, cuando un usuario con
permisos de anulación ejecuta la acción de eliminar dicha operación,
entonces el sistema la anula directamente, sin requerir la eliminación
previa de operaciones posteriores; recalcula automáticamente los saldos
afectados, permitiendo que el resultado sea negativo; y la operación
anulada pasa al estado "Anulada", conservando trazabilidad completa y
motivo de anulación. No se requiere intervención de soporte.

**5.4 Reversión de saldo de efectivo** — Dado que se anula una operación
que tuvo impacto en el saldo de efectivo, cuando la anulación es confirmada
por un usuario con permisos, entonces el sistema revierte el impacto
original: si la operación era un ingreso, se resta el monto del saldo
actual; si era un egreso, se suma el monto al saldo actual. El resultado
puede ser un saldo negativo si el monto revertido supera el saldo
disponible actual. El saldo negativo se muestra en rojo y entre paréntesis
(ej. ($ 15.320,00)). Un saldo negativo no bloquea la anulación ni requiere
confirmación adicional.

**5.5 Reversión de activos y recálculo FIFO** — Dado que se anula una
operación de compra o venta de un activo dentro del período habilitado,
cuando la anulación es confirmada, entonces el sistema revierte el impacto
original sobre la tenencia: si era una compra, se resta la cantidad
adquirida; si era una venta, se suma la cantidad vendida. Si la reversión
genera stock negativo, se permite (mostrado en rojo entre paréntesis, ej.
(100 acciones YPF)). El sistema recalcula la composición FIFO de los lotes
restantes según los lotes efectivamente disponibles al momento de la
anulación. Si la anulación es parcial, el FIFO se recalcula eliminando solo
los lotes afectados por dicha anulación.

**5.6 Manejo de saldos negativos** — Dado que como resultado de una
anulación el saldo de efectivo o la tenencia de un activo queda en valor
negativo, cuando el sistema calcula el saldo o tenencia post-anulación,
entonces el saldo negativo se registra y almacena como valor negativo,
visualizado en rojo y entre paréntesis. Comportamiento operativo: bloqueado
para egresos de fondos y ventas de activos sobre la posición negativa;
permitido para ingresos de fondos y compras de activos que reviertan o
compensen el negativo. El estado negativo no genera restricciones sobre
otras cuentas o instrumentos del usuario. Se registra en auditoría todo
evento que originó el saldo negativo. Las rentabilidades afectadas se
reversan al momento de la anulación. Un saldo ya negativo no genera saldo
"disponible" para nuevas operaciones de egreso, salvo la excepción de las
operaciones pendientes creadas precedentemente a la anulación.

**5.7 Absorción vía saldo negativo con operaciones pendientes intactas** —
Dado que existe una operación registrada con una o más operaciones en
estado "Pendiente de Control" o "Pendiente Banco" que mantienen fondos o
activos comprometidos sobre ella, cuando un usuario anula la operación
registrada que constituye la base económica del compromiso, entonces el
sistema permite la anulación sin bloquearla por dependencias, mantiene
intactas las operaciones pendientes y sus compromisos previos, recalcula el
saldo total y disponible post-anulación (absorbiendo el impacto mediante
saldo negativo si corresponde cuando la pendiente sea aprobada), y registra
en auditoría la anulación, la persistencia de las pendientes relacionadas y
el resultado final de la posición.

**5.8 Permisos para rechazar operaciones (roles junior y senior)** — Dado
que un usuario con rol junior o senior tiene una operación propia en estado
"Pendiente Control" o "Pendiente Bancaria", cuando decide rechazarla para
corregir un error, entonces el sistema permite el rechazo siempre que la
operación haya sido ingresada por el mismo usuario, sin requerir
intervención de un rol autorizante. La operación pasa al estado
"Rechazada" con trazabilidad completa. No se generan impactos en tenencias
ni saldos, dado que las operaciones pendientes no están registradas.

## 6. Impacto en UI / Interfaz de usuario

### 6.1 Visualización de saldos y tenencias negativas

| Elemento | Valor positivo | Valor negativo (en rojo) |
| --- | --- | --- |
| Saldo efectivo | $ 25.000,00 | ($ 8.500,00) |
| Tenencia de activo | 1.200 acciones YPF | (250) Acciones YPF |
| Saldo disponible | $ 12.000,00 | ($ 3.000,00) — bloquea egresos |
| Rentabilidad | Positivo o negativo | Valor negativo admitido, en rojo |

### 6.2 Mensajes del sistema

| Situación | Mensaje en UI |
| --- | --- |
| Período cerrado | "El período seleccionado se encuentra cerrado. Contacte al administrador para habilitar el período correspondiente." |
| Egreso bloqueado por saldo negativo | "No es posible realizar egresos de fondos con saldo negativo. Realice un ingreso para compensar el saldo antes de continuar." |
| Motivo anulación | Campo de descripción, máximo 250 caracteres (aviso si se supera el máximo) |
| Operación ya anulada | Ícono de tacho deshabilitado |

## 7. Flujos de usuario

### 7.1 Camino feliz — Anulación simple

1. El usuario ingresa a la tabla de operaciones.
2. Selecciona la operación a anular (cualquiera dentro del período
   habilitado).
3. Ejecuta la acción "Anular".
4. Confirma y escribe el motivo, que persiste en el detalle de la
   operación.
5. El sistema anula la operación, revierte saldos en tiempo real y
   actualiza tenencias, rentabilidad, inventario y dashboards.
6. La operación queda en estado "Anulada".

### 7.2 Camino feliz — Anulación masiva

1. El usuario selecciona múltiples operaciones mediante checkbox.
2. Ejecuta la acción "Anular".
3. Confirma y escribe el motivo.
4. El sistema procesa el lote completo y actualiza en tiempo real.

### 7.3 Caminos alternativos / de abandono

- Período cerrado: el sistema bloquea la acción y muestra el mensaje
  correspondiente.
- El usuario no tiene rol habilitado para anular, o el monto escapa del
  monto habilitado según la matriz de autorización.
- Egreso bloqueado por saldo negativo: se muestra mensaje orientando a
  realizar un ingreso primero.
- Operación ya anulada: el sistema informa el estado actual e impide la
  re-anulación.
- Error en recálculo de saldos: el sistema revierte la operación y
  mantiene los saldos previos, mostrando un mensaje de error técnico.

## 8. Documentos relacionados

| Documento | Descripción / Relación |
| --- | --- |
| PRD-TAB-001 | Tabla de Operaciones — Define la estructura de la tabla, estados, tabs y columnas |
| PRD-TAB-001 Congelamiento de Fondos | Reglas de reserva de fondos y activos en operaciones pendientes; interacción directa con el modelo de saldos negativos |
| PRD-PRM-008 Cartas | Flujo de cartas para operaciones Pendiente Banco; relacionado con el estado previo al registro definitivo |
