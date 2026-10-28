"""
categories/computadores_schema.py

Declaración de datos de la categoría "Computadores" para el wizard
genérico (ver plan: Fase 1 solo Computadores). Los ~20 atributos y sus
tipos migran acá desde `agents/ficha_agent.py::FILLABLE_ATTRIBUTES`
(línea 33) -- ese dict sigue existiendo tal cual para el chat legacy
(`agents/computadores_rules.py` es la pieza compartida entre ambos, ver
ese módulo), esto es la misma información pero en forma declarativa para
que el wizard la pueda renderizar sin código nuevo por atributo.

`core_order` refleja lo que ya hace cumplir `price_service.py`
(`_RELAX_ORDER`, línea 58): tipo_equipo, total_ram_gb y
total_almacenamiento_gb nunca se relajan en la búsqueda de precio -- son
los atributos que de verdad determinan el precio, y por eso son los
"core" que bloquean el precio completo hasta llenarse. El procesador es
core también, pero puede resolverse por dos vías distintas (línea/familia
en lista, o modelo exacto) -- de ahí la key lógica de paso
"linea_procesador_o_procesador", que no es un atributo real sino que el
agente de extracción resuelve a uno de los dos según lo que haya dicho el
usuario (ver la "regla de oro" en el prompt original: usar línea salvo
que el usuario dé un modelo exacto).
"""

from categories.schema import AttributeSchema, CategorySchema, register

SCHEMA = CategorySchema(
    slug="computadores",
    categoria_db="Computadores",
    label="Computadores",
    supports_convenio_marco=True,
    core_order=[
        "tipo_equipo",
        "linea_procesador_o_procesador",
        "total_ram_gb",
        "total_almacenamiento_gb",
    ],
    attributes={
        "tipo_equipo": AttributeSchema(
            key="tipo_equipo", label="Tipo de equipo", type="enum",
            description="Factor de forma del equipo",
            values=["Laptop", "AIO", "Desktop", "Otro"],
            core=True,
        ),
        "linea_procesador": AttributeSchema(
            key="linea_procesador", label="Procesador (línea)", type="enum_multi",
            description='Familia del procesador sin número de modelo (ej. "Intel Core i5", "AMD Ryzen 7"). Admite varias opciones a la vez.',
            values_source="dictionary",
            core=True,
        ),
        "procesador_principal": AttributeSchema(
            key="procesador_principal", label="Procesador (modelo exacto)", type="dict",
            description="Modelo completo del procesador (ej: Intel Core i5-1335U, AMD Ryzen 5 7530U)",
            values_source="dictionary",
            optional_detail=True,
        ),
        "total_ram_gb": AttributeSchema(
            key="total_ram_gb", label="Memoria RAM", type="numeric_range",
            description="Memoria RAM total en GB",
            unit="GB", min=4, max=64, allowed_steps=[4, 8, 16, 32, 64],
            core=True,
        ),
        "total_almacenamiento_gb": AttributeSchema(
            key="total_almacenamiento_gb", label="Almacenamiento", type="numeric_range",
            description="Almacenamiento total en GB",
            unit="GB", min=128, max=4000,
            allowed_steps=[128, 256, 512, 1000, 2000, 4000],
            core=True,
        ),
        "marca": AttributeSchema(
            key="marca", label="Marca", type="dict",
            description="Fabricante del equipo (ej: HP, Dell, Lenovo, Apple, Acer) -- nunca el fabricante del procesador",
            values_source="dictionary",
            optional_detail=True,
        ),
        "tecnologia_ram": AttributeSchema(
            key="tecnologia_ram", label="Tecnología RAM", type="enum",
            description="Estándar de memoria RAM",
            values=["DDR5", "DDR4", "LPDDR5X", "LPDDR5", "LPDDR4X", "LPDDR4", "DDR3"],
            optional_detail=True,
        ),
        "tecnologia_disco_principal": AttributeSchema(
            key="tecnologia_disco_principal", label="Tecnología de disco", type="enum",
            description="Tecnología del disco de mayor jerarquía",
            values=["NVMe SSD", "SATA SSD", "SSD", "HDD", "eMMC", "mSATA"],
            optional_detail=True,
        ),
        "tipo_configuracion_discos": AttributeSchema(
            key="tipo_configuracion_discos", label="Configuración de discos", type="enum",
            description="Combinación de discos instalados",
            values=["solo SSD", "SSD+HDD", "solo HDD", "otro"],
            optional_detail=True,
        ),
        "tiene_gpu_dedicada": AttributeSchema(
            key="tiene_gpu_dedicada", label="¿GPU dedicada?", type="boolean",
            description="¿El equipo tiene GPU discreta?",
            optional_detail=True,
        ),
        "gpu_dedicada_nombre": AttributeSchema(
            key="gpu_dedicada_nombre", label="Modelo de GPU dedicada", type="dict",
            description="Nombre completo de la GPU dedicada (solo si tiene_gpu_dedicada=true)",
            values_source="dictionary",
            optional_detail=True,
        ),
        "sistema_operativo": AttributeSchema(
            key="sistema_operativo", label="Sistema operativo", type="dict",
            description="Sistema operativo preinstalado",
            values_source="dictionary",
            optional_detail=True,
        ),
        "wifi_generacion": AttributeSchema(
            key="wifi_generacion", label="Wi-Fi", type="enum",
            description="Estándar Wi-Fi más alto disponible",
            values=["Wi-Fi 7", "Wi-Fi 6E", "Wi-Fi 6", "Wi-Fi 5", "Wi-Fi 4"],
            optional_detail=True,
        ),
        "pantalla_pulgadas": AttributeSchema(
            key="pantalla_pulgadas", label="Tamaño de pantalla", type="numeric_range",
            description="Tamaño diagonal de pantalla en pulgadas",
            unit='"', min=11.6, max=32,
            allowed_steps=[11.6, 13.3, 14, 15.6, 16, 17.3, 21.5, 23.8, 27, 32],
            optional_detail=True,
        ),
        "nombre_modelo": AttributeSchema(
            key="nombre_modelo", label="Nombre del modelo", type="free",
            description="Nombre comercial completo del modelo (opcional)",
            optional_detail=True,
        ),
    },
)

register(SCHEMA)
