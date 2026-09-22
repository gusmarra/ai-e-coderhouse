# PRD-XXX-000 – Cuentas destino y origen

## Propósito del documento

Este documento detalla la interacción entre cuentas comitentes y cuentas
bancarias cuando el usuario realiza una operación donde el dinero liquidado
tras una venta, o el dinero que origina una compra de activos, se acredite o
debite de una cuenta bancaria, y no del saldo disponible dentro de esa cuenta
comitente.

## El problema

El problema de los usuarios es que si desean registrar la suscripción de un
FCI, no basta con registrar una operación, sino que deben registrar también
una acreditación ficticia a la cuenta comitente.

## El impacto

- 100% de nuestros clientes (actuales y potenciales) se encuentran impactados.
- El esfuerzo operativo de nuestros clientes aumenta en un 100% ya que se
  duplican las operaciones registradas para obtener el mismo resultado.
- El problema es bloqueante para la contratación, según conversación testigo
  con Garantizar SGR.

## Solución propuesta

La solución introduce únicamente la posibilidad de indicar el origen
(compras/suscripciones) o el destino (ventas/rescates) de los fondos
asociados a una operación. Cuando el usuario seleccione una cuenta,
Alphinance omitirá la generación del movimiento de efectivo sobre la cuenta
comitente y asociará la operación a la cuenta seleccionada. No se modifican
los circuitos de liquidación, conciliación bancaria ni la operatoria de
transferencias entre cuentas, los cuales permanecerán fuera del alcance de
esta implementación.

## Objetivos

Este desarrollo es exitoso si un usuario puede, en una única operación y
visitando un único formulario:

- Adquirir cualquier activo financiero implementado en Alphinance con fondos
  de una de sus cuentas bancarias asociadas al cliente, sin necesidad de
  utilizar fondos de la cuenta comitente donde será custodiado dicho activo.
- Vender cualquier activo financiero implementado en Alphinance registrando
  la acreditación del efectivo resultante directamente en una de sus cuentas
  bancarias, sin que dicho efectivo impacte en ningún momento el saldo de la
  cuenta comitente donde estaba custodiado ese activo.
- Adquirir o vender cualquier activo afectando los saldos de la comitente,
  tal cual el sistema se encuentra implementado a día de hoy.
- Respetando: permisos por monto, permisos de visualización/gestión de las
  cuentas, permisos de visualización/gestión de la empresa.

## Alcance funcional

En los formularios se mantendrá:

- Dropdown select para seleccionar la cuenta de origen de fondos (hoy
  llamada cuenta comitente).
- Dropdown select para seleccionar la cartera de origen de fondos.

Se agregará una nueva sección antes de comisiones llamada "cuenta destino":

- Dropdown select para seleccionar la cuenta de destino de fondos.
- Dropdown select para seleccionar la cartera de destino de fondos.

Las "cuentas origen" deberán desplegar todas las cuentas creadas por el
usuario para la empresa (tanto las C.Cmte. como las C.Cta.) con el formato
de ejemplo "C.Cmte. 1234 - Galicia Securities S.A.U.". Si `company_broker`
es del tipo broker se usará el prefijo "C.Cmte."; si es banco se usará
"C.Cta.". Además existirá la opción "Externo a Alphinance", que habilita 4
campos obligatorios:

- Tipo de entidad originaria (Banco o Broker Cuenta).
- Nombre de la entidad originaria (tipeable, letras/números/símbolos, hasta
  50 caracteres).
- Tipo de cuenta originaria (se autocompleta: cuenta corriente para banco,
  cuenta broker para broker).
- Número de cuenta originaria (tipeable, números/puntos/guion/barra, hasta
  50 caracteres).

Se deberá validar siempre el saldo de efectivo disponible en la cartera de
origen cuando se trate de un egreso de fondos (compras, suscripciones,
transferencias de efectivo), excepto en la opción "Externo a Alphinance". En
los egresos de activos (ventas, rescates) se validarán las cantidades
disponibles del activo.

Las "cuentas destino" siguen la misma lógica de listado y de opción "Externo
a Alphinance" (con los mismos 4 campos obligatorios). Se valida el saldo de
efectivo disponible cuando se trate de un egreso de fondos, salvo la opción
externa, y las cantidades disponibles del activo cuando se trate de egresos
de activos.

### Excepciones para las cuentas de destino

1. En formularios de egreso de activos (que acredita efectivo en la "cuenta
   destino"), el sistema debe validar que la cuenta tenga la MONEDA
   HABILITADA para recibir los fondos; si no, debe mostrar: "No posee la
   moneda habilitada en la cuenta seleccionada. Por favor habilitar desde
   Brokers."
2. En esos mismos formularios, además de la moneda habilitada, debe
   validarse que exista balance inicial vigente en la FECHA ingresada en el
   formulario; si no, debe mostrar: "El balance inicial es posterior a la
   fecha seleccionada para dicha moneda."
3. En formularios de adquisición de activos (compras, suscripciones), más
   allá de la opción externa, solo deben mostrarse las cuentas habilitadas
   para contener dicho activo (por ejemplo, para Plazo Fijo se muestran las
   cuentas corrientes; para el resto de los activos, las comitentes; para
   forex puede aplicar tanto cuenta corriente como cuenta comitente).
4. Se pide tanto "cuenta destino" como "cartera destino". La cuenta destino
   siempre será obligatoria; la cartera destino será obligatoria solo en los
   formularios de adquisición de activos, no en los de egresos de activos.
5. No se podrá seleccionar "Externo a Alphinance" en la cuenta destino si en
   la cuenta origen ya fue seleccionada esa opción, y viceversa.

### Pantallas afectadas

Formularios de compra de: Opciones, Suscripción de FCI, Acciones, Bonos,
Venture Capital (aporte de capital, contrato inicial), Plazo Fijo.

Formularios de venta de: Opciones, Rescate de FCI, Acciones, Bonos, Venture
Capital (venta).

Pantalla de Operaciones (`https://app.alphinance.com/investments/operations`):
campo cuenta, campo cuenta destino, campo cuenta origen.

## Fuera de alcance

Préstamos, Divisas/Forex, Venture Capital (revaluación), Movimientos de
efectivo.

## Reglas funcionales

1. No se especifican usuarios, roles ni permisos adicionales a los
   existentes.
2. Los permisos parametrizados para cada usuario deben continuar en efecto.
3. La cuenta/cartera destino por defecto se seleccionará con la cuenta
   origen / cartera seleccionada.
4. El dropdown sólo mostrará cuentas permitidas para el usuario.

## Casos de uso

- Usuario quiere registrar la suscripción de un FCI utilizando saldo de la
  cuenta comitente.
- Usuario quiere registrar el rescate de un FCI con acreditación a la cuenta
  comitente.
- Usuario quiere registrar la suscripción de un FCI utilizando el saldo de
  la cuenta corriente bancaria.
- Usuario quiere registrar el rescate de un FCI con acreditación a la cuenta
  corriente bancaria.

## Tablas afectadas

| Tabla / Entidad | Acción | Descripción |
| --- | --- | --- |
| Operaciones | Operación exitosa de compra | Afecta el campo "Cuenta Origen" |
| Operaciones | Operación exitosa de venta | Afecta el campo "Cuenta Destino" |

Impacto en tablas de: Operaciones, Brokers (balances), Carteras (balances),
Inventario, Tenencia, Rentabilidad Desagregada y, por lo tanto, Consolidada.

## Visualización en la tabla de Operaciones

- Cuando la cuenta origen es la misma que la cuenta destino, la operación se
  ve reflejada en una sola línea dentro de la tabla de operaciones, excepto
  las operaciones compuestas (ver PRD-INV-001 - Operaciones Compuestas).
- Cuando la operación involucra una cuenta "Externa a Alphinance" (por
  origen o por destino), también se ve reflejada en una sola línea, salvo
  que además sea una operación compuesta.
- Cuando la cuenta origen difiere de la cuenta destino (y ninguna es
  externa), la operación se refleja en 2 líneas diferentes: una por el
  egreso y otra por el ingreso.

## Escenarios mínimos de testing

**Camino feliz 1**: empresa con Comitente #1, Comitente #2, Cuenta Bancaria
#3, Cuenta Bancaria #4. Cuenta #1 con saldo de efectivo de $1.000; Cuenta #3
con saldo de efectivo positivo de $1.000. El usuario suscribe FCI por 1000
cuotapartes de precio $0,5 (monto total $500) con Cuenta Origen = #3.
Resultado esperado: operación exitosa; en la tabla de Operaciones, Cuenta #1
aparece con cuenta origen #3; tenencia en cuenta #1: $1000 y 1000
cuotapartes; tenencia en cuenta #3: $500.

**Camino feliz 2**: mismas cuentas, con Cuenta #1 con $1.000 de efectivo y
1000 cuotapartes ($0,5 c/u, $500 total); Cuenta #3 con $5.000. El usuario
rescata FCI por 200 cuotapartes de precio $0,5 (monto total $100) con
Cuenta Destino = #3. Resultado esperado: operación exitosa; Cuenta #1
aparece con cuenta destino #3; tenencia en cuenta #1: $1000 y 800
cuotapartes; tenencia en cuenta #3: $600.

## Deuda técnica / funcional — evoluciones potenciales

- Persistencia de cuenta bancaria elegida según ID de instrumento o cuenta.
- Creación de tipo de cuenta "cuenta cuotapartista".
