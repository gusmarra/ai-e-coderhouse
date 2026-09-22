# Operaciones Compuestas — Tabla de Operaciones (Alphinance)

Documento funcional para integración al PRD-TAB-001 Tabla de Operaciones.

## 1. Definición

Se define como operación compuesta a toda transacción que, por su naturaleza
operativa, genera dos o más registros (líneas) en la tabla de operaciones,
los cuales representan un único evento económico y comparten un mismo
número de transacción (campo #, autogenerado por el sistema). Estas líneas
deben considerarse transacciones vinculadas, permitiendo su trazabilidad,
visualización agrupada y anulación conjunta.

El identificador compartido entre las líneas de una operación compuesta es
el número de transacción (campo # de la tabla), no el ID del instrumento.
Cada línea puede referenciar un instrumento distinto y conserva su propio ID
de instrumento.

## 2. Tipología general

Las operaciones compuestas se clasifican en dos grandes grupos:

- **Transferencias internas**: movimientos dentro del ecosistema gestionado
  en la plataforma. Involucran cuentas, brokers o carteras registradas. No
  modifican el patrimonio total consolidado (salvo comisiones en cuenta de
  origen).
- **Inversiones multi-instrumento**: transacciones en las que un único
  evento económico implica la participación simultánea de dos o más activos
  financieros como origen, destino o contraparte.

## 3. Casos contemplados

### 3.1 Transferencia de fondos entre brokers

Movimiento de efectivo desde una cuenta en Broker A hacia una cuenta en
Broker B.

| Tipo de línea | Descripción |
| --- | --- |
| Egreso | Cuenta Broker A — salida de efectivo por el monto transferido |
| Ingreso | Cuenta Broker B — entrada de efectivo por el mismo monto |

No modifica el patrimonio total consolidado. Las comisiones, si aplican, se
registran en la línea de egreso (cuenta origen).

### 3.2 Transferencia entre carteras de distintas cuentas / brokers

Movimiento de activos o efectivo desde una cartera/cuenta hacia otra.

| Tipo de línea | Descripción |
| --- | --- |
| Egreso | Cartera/Cuenta A — salida del activo o efectivo |
| Ingreso | Cartera/Cuenta B — entrada del mismo activo o efectivo |

No modifica el patrimonio total consolidado. Las comisiones, si aplican, se
registran en la línea de egreso (cuenta origen).

### 3.3 FCI — Suscripción en especie

Suscripción a un Fondo Común de Inversión mediante la entrega de un
instrumento (ej. bono) como contraparte, sin intervención de efectivo.

| Tipo de línea | Descripción |
| --- | --- |
| Ingreso | Cuotapartes del FCI suscripto |
| Egreso | Instrumento entregado como contraparte (ej. bono) |

El valor del instrumento entregado debe valuarse al precio de mercado
vigente al momento de la operación, a efectos del cálculo de la
equivalencia con las cuotapartes recibidas. Será un input a incluir por el
usuario.

### 3.4 Forex / Divisas

Conversión entre dos monedas (ej. USD → ARS, EUR → USD). Operación
compuesta nativa: siempre involucra exactamente dos instrumentos (monedas)
vinculados por el tipo de cambio aplicado.

| Tipo de línea | Descripción |
| --- | --- |
| Egreso | Moneda origen — monto vendido/convertido |
| Ingreso | Moneda destino — monto recibido al tipo de cambio operado |

Siempre genera exactamente 2 líneas. Las comisiones e impuestos se
registran embebidos en el monto de la línea de egreso, no como una tercera
línea separada. Se deben guardar los tipos de cambio aplicados.

### 3.5 Ejercicio de opciones (pendiente de validación)

Incluido como referencia funcional; requiere revisión antes de su
implementación.

**3.5.1 Ejercicio de un Call**: el tenedor de un call ejerce su derecho a
comprar el subyacente al precio de strike. El multiplicador determina la
cantidad de subyacente recibido.

| Tipo de línea | Descripción |
| --- | --- |
| Egreso (1) | Cierre de la posición de opción (call ejercido) |
| Egreso (2) | Pago del precio de strike × multiplicador (efectivo) |
| Ingreso | Activo subyacente recibido — cantidad = multiplicador |

**3.5.2 Ejercicio de un Put**: el tenedor de un put ejerce su derecho a
vender el subyacente al precio de strike.

| Tipo de línea | Descripción |
| --- | --- |
| Egreso (1) | Cierre de la posición de opción (put ejercido) |
| Egreso (2) | Entrega del activo subyacente — cantidad = multiplicador |
| Ingreso | Cobro del precio de strike × multiplicador (efectivo) |

**3.5.3 Ejercicio por excedentes (cash settlement)**: el ejercicio se
liquida por diferencia en efectivo, sin entrega física del subyacente.

| Tipo de línea | Descripción |
| --- | --- |
| Egreso | Cierre de la posición de opción |
| Ingreso | Acreditación de efectivo: diferencia entre precio de mercado y strike × multiplicador |

En todos los casos de ejercicio, el cómputo del costo de adquisición del
subyacente recibido debe tomar en cuenta la prima pagada originalmente por
la opción.

### 3.6 Asignación de opciones (pendiente de validación)

La asignación es el espejo del ejercicio: ocurre cuando el lanzador
(vendedor) de una opción es notificado de que el comprador decidió
ejercerla, quedando obligado a cumplir con las condiciones del contrato. A
diferencia del ejercicio, que es una decisión del tenedor, la asignación es
involuntaria para quien la recibe.

**3.6.1 Asignación de un Call vendido**: el lanzador es obligado a entregar
el subyacente al precio de strike, y recibe el pago en efectivo.

| Tipo de línea | Descripción |
| --- | --- |
| Egreso (1) | Cierre de la posición vendida de opción (call asignado) |
| Egreso (2) | Entrega del activo subyacente — cantidad = multiplicador |
| Ingreso | Cobro del precio de strike × multiplicador (efectivo) |

**3.6.2 Asignación de un Put vendido**: el lanzador es obligado a comprar el
subyacente al precio de strike, pagando en efectivo y recibiendo los
activos del comprador.

| Tipo de línea | Descripción |
| --- | --- |
| Egreso (1) | Cierre de la posición vendida de opción (put asignado) |
| Egreso (2) | Pago del precio de strike × multiplicador (efectivo) |
| Ingreso | Recepción del activo subyacente — cantidad = multiplicador |

La prima cobrada originalmente al lanzar la opción queda registrada en la
operación de apertura (lanzamiento).

## 4. Consideraciones operativas

### 4.1 Identificador compartido

Todas las líneas generadas por una operación compuesta deben compartir el
mismo número de transacción (campo # de la tabla), mantener la misma fecha
de concertación en todas las líneas, y conservar su propio ID de
instrumento individual (ticker, ISIN, código de divisa, etc.).

### 4.2 Ordenamiento visual en la UI

Primera línea: el hecho económico principal (el instrumento que el usuario
adquiere o recibe). Líneas siguientes: las contrapartidas (lo que se
entrega, paga o debita).

| Operación | Orden de líneas en UI |
| --- | --- |
| Transferencia de fondos | 1° Ingreso en cuenta destino / 2° Egreso en cuenta origen |
| FCI suscripción en especie | 1° Ingreso de cuotapartes / 2° Egreso del instrumento entregado |
| Forex | 1° Ingreso en moneda destino / 2° Egreso en moneda origen |
| Ejercicio de Call | 1° Ingreso del subyacente / 2° Egreso de efectivo (strike) / 3° Cierre de opción |
| Ejercicio de Put | 1° Ingreso de efectivo (strike) / 2° Egreso del subyacente / 3° Cierre de opción |
| Ejercicio por excedentes | 1° Ingreso de efectivo / 2° Cierre de opción |

### 4.3 Anulación de operaciones compuestas

La unidad mínima de anulación es la operación compuesta completa. No es
posible anular una línea individual dentro de una operación compuesta. La
anulación debe impactar todas las líneas vinculadas simultáneamente,
restituir saldos y posiciones de forma íntegra en todas las cuentas,
instrumentos y carteras afectados, y aplicar las reglas de saldos negativos
del PRD-INV-001 si la restitución genera posiciones deficitarias.

### 4.4 Interacción con el módulo de Congelamiento de Fondos

Cuando una operación compuesta está en estado "Pendiente de Control" o
"Pendiente Banco", el congelamiento se aplica sobre todas sus líneas en
simultáneo: se congela el efectivo comprometido en la línea de egreso de
efectivo, se congela la cantidad de activos comprometidos en la línea de
egreso de instrumentos, el ingreso esperado no se acredita hasta la
aprobación, y si la operación es rechazada o vence el plazo de 72 hs, se
liberan todos los compromisos de todas las líneas en simultáneo.

## 5. Escalabilidad del modelo

El modelo de operaciones compuestas debe soportar más de dos líneas por
operación, según la complejidad del evento financiero. El ejercicio de
opciones ya es un ejemplo de operación con 3 líneas.

**Ejemplo futuro — suscripción mixta de FCI (efectivo + especie)**: una
suscripción que combina pago parcial en efectivo y entrega de uno o más
instrumentos como contraparte generaría:

| Tipo de línea | Descripción |
| --- | --- |
| Ingreso | Cuotapartes del FCI suscripto |
| Egreso (1) | Efectivo aportado |
| Egreso (2) | Instrumento A entregado como contraparte |
| Egreso (3) | Instrumento B entregado como contraparte (si aplica) |

Las operaciones compuestas constituyen un patrón base del modelo, no una
excepción. La tabla de operaciones debe representar la actividad real de la
empresa, incluso cuando una operación implique múltiples impactos
simultáneos. Se prioriza la trazabilidad completa sobre la simplificación
visual.

## 6. Resumen de casos por cantidad de líneas

| Operación | Líneas | Estado |
| --- | --- | --- |
| Transferencia de fondos entre brokers | 2 | Definido |
| Transferencia entre carteras/cuentas | 2 | Definido |
| FCI — Suscripción en especie | 2 | Definido |
| Forex / Divisas | 2 | Definido |
| Ejercicio de Call (opción) | 3 | A validar |
| Ejercicio de Put (opción) | 3 | A validar |
| Ejercicio por excedentes (opción) | 2 | A validar |
| Asignación de Call vendido (opción) | 3 | A validar |
| Asignación de Put vendido (opción) | 3 | A validar |
| Suscripción mixta FCI (efectivo + especie) | 3+ | Futuro |
