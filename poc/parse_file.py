"""
Парсер PDF и документов с коррекцией текста на основе LLM

Этот модуль предоставляет комплексное решение для парсинга документов, поддерживающее множество
форматов (PDF, DOCX, HTML, PPTX, ASCIIDOC, MD) с возможностями OCR и коррекции текста на основе
языковых моделей для повышения точности.

Автор: [Гладышев Виталий, Авдеев Леонид]
Версия: 1.0.0
Лицензия: IDT NDA
"""

import codecs
import os
import re
from pathlib import Path
from typing import List, Dict, Optional

import httpx
import pandas as pd
from docling.datamodel.accelerator_options import AcceleratorOptions
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions, PaginatedPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption, WordFormatOption, HTMLFormatOption, \
                                        AsciiDocFormatOption, XMLJatsFormatOption
from docling_core.types.doc.document import PictureItem, TableItem, TextItem, TitleItem, ListItem, CodeItem, SectionHeaderItem

from .config import Config
from .service_classes import DocumentConfig


class DocumentParser:
    """
    Продвинутый парсер документов с возможностями OCR и коррекцией текста на основе LLM.
    
    Этот класс предоставляет комплексную функциональность парсинга документов для множества форматов
    включая PDF, DOCX, HTML, PPTX, ASCIIDOC и Markdown. Он включает OCR обработку,
    извлечение таблиц, обработку изображений и интеллектуальную коррекцию текста с использованием LLM моделей.
    
    Атрибуты:
        document (pd.DataFrame): DataFrame, содержащий обработанное содержимое документа
        llm (LLM): Экземпляр языковой модели для коррекции текста
        config (DocumentConfig): Настройки конфигурации для обработки документов
        parser (DocumentConverter): Экземпляр конвертера документов Docling
        
    Пример:
        >>> parser = DocumentParser("http://localhost:9090")
        >>> df = parser.parse_file("document.pdf", "output/images")
        >>> parser.clean_text()
        >>> parser.create_html_files("output/html")
    """
    
    def __init__(self, llm, server_url: str, ids_service, document_config: Optional[DocumentConfig] = None):
        """
        Инициализация DocumentParser.
        
        Args:
            server_url (str): URL сервера генерации ID
            document_config (DocumentConfig, optional): Настройки конфигурации. По умолчанию None.
        """
        self.server_url = server_url
        self.document_config = document_config or DocumentConfig()
        self.logger = Config().logger
        
        # Инициализация структур данных
        self.document = pd.DataFrame(columns=['Header', 'Text', 'CleanText', 'HTML', 'Tables', 'Images'])
        self.images_data: List[Dict] = []
        self.tables_data: List[Dict] = []
        self.file_name: str = ""
        self.is_pdf: bool = False
        self.unique_id = ""
        self.output_dir = Path("")
        self.llm = llm
        self.ids_service = ids_service
        
        # Инициализация парсера
        self._init_parser()
    
    def _init_parser(self) -> None:
        """Инициализация парсера документов Docling с настроенными опциями."""
        try:
            # Настройка опций ускорителя
            accelerator_options = AcceleratorOptions(
                num_threads=self.document_config.num_threads,
                device=self.document_config.device
            )
            
            # Настройка опций пайплайна
            pipe_opts = PaginatedPipelineOptions()
            pipe_opts.accelerator_options = accelerator_options
            pipe_opts.images_scale = self.document_config.images_scale
            pipe_opts.generate_page_images = True
            pipe_opts.generate_picture_images = True
            
            # Настройка PDF-специфичных опций
            pdf_pipeline_options = PdfPipelineOptions()
            pdf_pipeline_options.do_ocr = True
            pdf_pipeline_options.ocr_options.lang = self.document_config.ocr_languages
            pdf_pipeline_options.ocr_options.force_full_page_ocr = False
            pdf_pipeline_options.do_table_structure = True
            pdf_pipeline_options.table_structure_options.do_cell_matching = True
            pdf_pipeline_options.generate_picture_images = True
            pdf_pipeline_options.do_picture_classification = False
            pdf_pipeline_options.do_picture_description = False
            pdf_pipeline_options.images_scale = self.document_config.images_scale
            pdf_pipeline_options.accelerator_options = accelerator_options
            
            # Инициализация конвертера документов
            self.parser = DocumentConverter(
                allowed_formats=[
                    InputFormat.PDF,
                    InputFormat.DOCX,
                    InputFormat.HTML,
                    InputFormat.PPTX,
                    InputFormat.ASCIIDOC,
                    InputFormat.MD,
                    InputFormat.XML_USPTO
                ],
                format_options={
                    InputFormat.PDF: PdfFormatOption(pipeline_options=pdf_pipeline_options),
                    InputFormat.DOCX: WordFormatOption(pipeline_options=pipe_opts),
                    InputFormat.HTML: HTMLFormatOption(pipeline_options=pipe_opts),
                    InputFormat.ASCIIDOC: AsciiDocFormatOption(pipeline_options=pipe_opts),
                    InputFormat.XML_USPTO: XMLJatsFormatOption(pipeline_options=pipe_opts),
                }
            )
            self.logger.info("Парсер документов успешно инициализирован")
        except Exception as e:
            self.logger.error(f"Не удалось инициализировать парсер документов: {e}")
            raise
    
    def parse_file(self, file_path: str, output_dir: Optional[Path] = None) -> pd.DataFrame:
        """
        Парсинг файла документа и извлечение его содержимого.
        
        Args:
            file_path (str): Путь к файлу документа
            output_dir (str, optional): Директория для сохранения извлеченных данных
            
        Returns:
            pd.DataFrame: DataFrame, содержащий обработанное содержимое документа
            
        Raises:
            FileNotFoundError: Если указанный файл не существует
            ValueError: Если формат файла не поддерживается
            Exception: Для других ошибок парсинга
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Файл не найден: {file_path}")
        
        self.logger.info(f"Парсинг документа: {file_path}")
        
        # Установка свойств файла
        self.file_name = Path(file_path).stem
        extension = Path(file_path).suffix.lower()
        self.is_pdf = extension == ".pdf"
        
        # Создание директории для извлеченных данных
        self.output_dir = output_dir or Path("./output")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        # Создание директории для изображений, если указана
        images_output_dir = self.output_dir / "images"
        images_output_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            # Парсинг документа с помощью Docling
            doc = self.parser.convert(file_path)
            self.logger.info(f"Статус документа: {doc.status}")

            # Извлечение структурированного содержимого
            chapters = self._extract_chapters_from_docling(doc, images_output_dir)
            
            # Заполнение DataFrame
            self._populate_dataframe(chapters)
            
            self.logger.info(f"Извлечено {len(self.document)} глав, "
                             f"{len(self.tables_data)} таблиц, "
                             f"{len(self.images_data)} изображений")
            
            # Сохранение результатов
            self.save(self.output_dir / Path(f"{self.file_name}.csv"))
            
            return self.document
            
        except Exception as e:
            self.logger.error(f"Не удалось обработать файл {file_path}: {e}")
            raise
    
    def _populate_dataframe(self, chapters: List[Dict]) -> None:
        """Заполнение основного DataFrame данными извлеченных глав."""
        for chapter in chapters:
            if len(chapter['text']) > 1:
                new_row = {
                    'Header': chapter['header'],
                    'Text': chapter['text'],
                    'CleanText': chapter['clean'],
                    'HTML': '',
                    'Tables': chapter.get('tables', []),
                    'Images': chapter.get('images', [])
                }
                self.document = pd.concat([
                    self.document,
                    pd.DataFrame([new_row], columns=self.document.columns)
                ], ignore_index=True)
    
    def _extract_chapters_from_docling(self, doc, images_output_dir: Optional[Path] = None) -> List[Dict]:
        """
        Извлечение глав и разделов из документа Docling.
        
        Args:
            doc: Объект документа Docling
            images_output_dir: Директория для сохранения изображений
            
        Returns:
            Список словарей глав
        """
        chapters = []
        current_chapter = {
            'header': "",
            'text': "",
            'clean': "",
            'tables': [],
            'images': []
        }
        chapter_counter = 0
        image_counter = 0
        table_counter = 0

        try:
            # Создание директории для изображений, если указана
            images_output_dir = images_output_dir or Path("./output/images")
            images_output_dir.mkdir(parents=True, exist_ok=True)
            # Создание директории с уникальным сгенерированным именем
            self.unique_id = self._generate_ids_sync(1)[0]
            images_folder = images_output_dir / Path(self.unique_id)
            images_folder.mkdir(parents=True, exist_ok=True)
            for element, _level in doc.document.iterate_items():

                if isinstance(element, (SectionHeaderItem, TitleItem)):
                    # Обработка заголовков
                    heading_text = self._clean_text_request(element.text) if self.is_pdf else element.text
                    if self._should_create_new_chapter(element):
                        if current_chapter and current_chapter['text'].strip():
                            chapters.append(current_chapter)
                        current_chapter = {
                            'header': heading_text,
                            'text': "",
                            'clean': "",
                            'tables': [],
                            'images': []
                        }
                        chapter_counter += 1
                    else:
                        current_chapter['text'] += f"\n\n{element.text}\n\n"
                        if self.is_pdf:
                            current_chapter['clean'] += f"\n\n{element.text}\n\n"
                
                elif isinstance(element, ListItem):
                    list_text = self._process_list(element)
                    current_chapter['text'] += list_text + "\n\n"
                    if self.is_pdf:
                        current_chapter['clean'] += list_text + "\n\n"
                
                elif isinstance(element, CodeItem):
                    current_chapter['text'] += element.text + "\n\n"
                    if self.is_pdf:
                        current_chapter['clean'] += element.text + "\n\n"
                
                elif isinstance(element, TextItem):
                    if self.is_pdf:
                        prepared_text = self._clean_text_request(element.text)
                        current_chapter['text'] += prepared_text + "\n\n"
                        current_chapter['clean'] += prepared_text + "\n\n"
                    else:
                        current_chapter['text'] += element.text + "\n\n"
                
                elif isinstance(element, TableItem):
                    table_counter += 1
                    table_data = self._process_table(doc,element, table_counter)
                    
                    if current_chapter:
                        current_chapter['tables'].append(table_data)
                        current_chapter['text'] += f"\n[Таблица {table_counter}]\n{table_data['text']}\n\n"
                        if self.is_pdf:
                            current_chapter['clean'] += f"\n[Таблица {table_counter}]\n{table_data['text']}\n\n"
                    
                    self.tables_data.append(table_data)
                
                elif isinstance(element, PictureItem):
                    image_counter += 1
                    image_data = self._process_image(element, doc, image_counter, images_folder)
                    
                    if current_chapter:
                        current_chapter['images'].append(image_data)
                        current_chapter['text'] += (
                            f"\n[Изображение {image_counter}: "
                            f"{image_data.get('caption', 'Без подписи')}]\n\n"
                        )
                        if self.is_pdf:
                            current_chapter['clean'] += (
                                f"\n[Изображение {image_counter}: "
                                f"{image_data.get('caption', 'Без подписи')}]\n\n"
                            )
                    
                    self.images_data.append(image_data)

            
            # Сохранение последней главы
            if current_chapter and current_chapter['text'].strip():
                chapters.append(current_chapter)
            
            # Обработка случая, когда главы не найдены
            if not chapters:
                chapters = self._create_fallback_chapter(doc)
            
            return chapters
            
        except Exception as e:
            self.logger.error(f"Ошибка при извлечении глав: {e}")
            raise
    
    def _should_create_new_chapter(self, element) -> bool:
        """Определение, должна ли быть создана новая глава на основе уровня элемента."""
        return (not hasattr(element, 'level') or 
                (hasattr(element, 'level') and element.level <= self.document_config.parceby_level))
    
    def _process_list(self, list_element) -> str:
        """Обработка элементов списка с опциональной LLM коррекцией."""
        prepared_text = (self._clean_text_request(list_element.text) 
                        if self.is_pdf else list_element.text)
        
        if list_element.marker:
            return f"{list_element.marker} {prepared_text}\n"
        return f"• {prepared_text}\n"
    
    def _process_table(self,doc, table_element, table_id: int) -> Dict:
        """
        Обработка элементов таблицы и извлечение их данных.
        
        Args:
            table_element: Элемент таблицы из Docling
            table_id: Уникальный идентификатор таблицы
            
        Returns:
            Словарь, содержащий данные таблицы
        """
        table_data = {
            'id': table_id,
            'caption': '',
            'html': '',
            'text': '',
            'dataframe': None
        }
        #self.logger.info(table_element)
        try:
            # Извлечение подписи
            if hasattr(table_element, 'captions'):
                table_data['caption'] = table_element.captions
            
            # Попытка экспорта как DataFrame
            if hasattr(table_element, 'export_to_dataframe'):
                df = table_element.export_to_dataframe(doc)
                if all(isinstance(col, int) for col in df.columns):
                    df_cleaned = df.iloc[1:].reset_index(drop=True)
                    df_cleaned.columns = df.iloc[1:, 0]
                    df_cleaned.index = df_cleaned.index + 1
                    table_data['html'] = df_cleaned.to_html(index=False, classes='docling-table',
                                                            table_id=f'table-{table_id}')
                    table_data['dataframe'] = df_cleaned
                    table_data['text'] = self._format_table_as_text(df_cleaned)
                else:
                    table_data['dataframe'] = df
                    table_data['html'] = df.to_html(
                        index=False,
                        classes='docling-table',
                        table_id=f'table-{table_id}'
                    )
                    table_data['text'] = self._format_table_as_text(df)
            
            # Альтернативный метод для данных таблицы
            elif hasattr(table_element, 'data'):
                table_data['text'] = self._format_table_data(table_element.data)
                table_data['html'] = self._format_table_as_html(table_element.data, table_id)
                
        except Exception as e:
            self.logger.warning(f"Ошибка при обработке таблицы {table_id}: {e}")
        #logger.info(table_data)
        return table_data
    
    def _process_image(self, image_element, doc, image_id: int, 
                      output_dir: Optional[Path] = None) -> Dict:
        """
        Обработка элементов изображения и их сохранение, если указана выходная директория.
        
        Args:
            image_element: Элемент изображения из Docling
            doc: Объект документа
            image_id: Уникальный идентификатор изображения
            output_dir: Директория для сохранения изображения
            
        Returns:
            Словарь, содержащий данные изображения
        """
        image_data = {
            'id': image_id,
            'caption': '',
            'path': '',
            'type': 'image'
        }
        
        try:
            # Создание директории для изображений, если указана
            images_output_dir = output_dir or Path("./output/images")
            images_output_dir.mkdir(parents=True, exist_ok=True)

            # Извлечение подписи
            if hasattr(image_element, 'captions'):
                image_data['caption'] = image_element.caption_text(doc.document)
            
            # Сохранение изображения, если указана выходная директория
            if images_output_dir:
                file_extension = 'png'
                filename = f"image_{image_id}.{file_extension}"
                filepath = images_output_dir / filename
                
                with open(filepath, 'wb') as fp:
                    image_element.get_image(doc.document).save(fp, file_extension)
                
                # Сохранение относительного пути
                if "images" in filepath.parts:
                    images_index = filepath.parts.index("images")
                    image_data['path'] = Path(*filepath.parts[images_index:])
                else:
                    image_data['path'] = filepath

                #logger.info(f"Изображение сохранено: {image_data['path']}")
                
        except Exception as e:
            self.logger.error(f"Ошибка при обработке изображения {image_id}: {e}")
        
        return image_data
    
    def _create_fallback_chapter(self, doc) -> List[Dict]:
        """Создание резервной главы, когда структурированные главы не найдены."""
        self.logger.warning("Главы не найдены, создается резервная глава")
        
        full_text = ""
        if hasattr(doc, 'export_to_markdown'):
            full_text = doc.export_to_markdown
        elif hasattr(doc, 'export_to_text'):
            full_text = doc.export_to_text
        
        if full_text:
            return [{
                'header': 'Документ',
                'text': full_text,
                'tables': self.tables_data,
                'images': self.images_data
            }]
        
        return []
    
    def _format_table_as_text(self, df: pd.DataFrame) -> str:
        """Форматирование DataFrame как форматированное текстовое представление."""
        if df is None or df.empty:
            return ""
        
        separator = "─" * 50
        formatted = f"\n{separator}\n"
        
        # Заголовки
        headers = " | ".join(str(col) for col in df.columns)
        formatted += f"| {headers} |\n{separator}\n"
        
        # Строки данных
        for _, row in df.iterrows():
            row_text = " | ".join(str(val) for val in row.values)
            formatted += f"| {row_text} |\n"
        
        formatted += f"{separator}\n"
        return formatted
    
    def _format_table_data(self, table_data: List[List]) -> str:
        """Форматирование данных таблицы из списка списков как текст."""
        if not table_data:
            return ""
        
        separator = "─" * 50
        formatted = f"\n{separator}\n"
        
        for row in table_data:
            formatted += "|"
            for cell in row:
                cell_text = str(cell) if cell is not None else ""
                formatted += f" {cell_text} |"
            formatted += "\n"
        
        formatted += f"{separator}\n"
        return formatted
    
    def _format_table_as_html(self, table_data: List[List], table_id: int) -> str:
        """Форматирование данных таблицы из списка списков как HTML."""
        if not table_data:
            return ""
        
        html = f'<table id="table-{table_id}" class="docling-table">\n'
        
        if len(table_data) > 0:
            # Заголовки
            html += "<thead><tr>"
            for cell in table_data[0]:
                html += f"<th>{cell if cell else ''}</th>"
            html += "</tr></thead>\n"
            
            # Строки данных
            if len(table_data) > 1:
                html += "<tbody>\n"
                for row in table_data[1:]:
                    html += "<tr>"
                    for cell in row:
                        html += f"<td>{cell if cell else ''}</td>"
                    html += "</tr>\n"
                html += "</tbody>\n"
        
        html += "</table>"
        return html
    
    def clean_text(self) -> pd.DataFrame:
        """
        Очистка всех текстов в колонке Text с помощью LLM.
        
        Returns:
            pd.DataFrame: DataFrame с очищенным текстом в колонке CleanText
        """
        data_size = self.document.shape[0]
        
        for i in range(data_size):
            if self.document.iloc[i]["Text"] and not self.document.iloc[i]["CleanText"]:
                try:
                    clean_text_result = self._clean_text_request(self.document.iloc[i]['Text'])
                    self.document.loc[i, 'CleanText'] = clean_text_result
                    self.logger.info(f"Очищен раздел {i+1}: {clean_text_result[:50]}...")
                except Exception as e:
                    self.logger.error(f"Ошибка при очистке текста для раздела {i+1}: {e}")
                    # Откат к исходному тексту
                    self.document.loc[i, 'CleanText'] = self.document.iloc[i]['Text']
        
        self.save(self.output_dir / Path(f"{self.file_name}.csv"))
        return self.document
    
    def _clean_text_request(self, text: str) -> str:
        """
        Очистка текста с помощью LLM с правильной обработкой ошибок.
        
        Args:
            text (str): Текст для очистки
            
        Returns:
            str: Очищенный текст или исходный текст при неудачной очистке
        """
        # Установка системного промпта
        self.llm.set_system_prompt(
            "Ты профессионально работаешь с текстом и выполняешь роль корректора. "
            "Исправляй опечатки, восстанавливай пропущенные пробелы и пунктуацию. "
            "Исправляй искажения текста после OCR, например, некорректное применение заглавных букв. "
            "Не сокращай смысл текста, а только исправляй опечатки и ошибки в тексте! "
            "Не добавляй лишних слов, выводи только результат"
        )
        
        prompt = (
            f"Исправь опечатки в документе. Вставь пробелы где необходимо. "
            f"Удали служебные символы. Убери лишние переносы строк. "
            f"Откорректируй текст после OCR, например, некорректное применение заглавных букв. "
            f"Не сокращай смысл текста, а только исправляй опечатки и ошибки в тексте! "
            f"Выведи только исправленный текст документа. "
            f"Документ:\n\n{text}"
        )
        
        try:
            clean_text = self.llm.generate_answer(prompt)
            return clean_text
        except Exception as e:
            self.logger.error(f"Ошибка при вызове LLM: {e}")
            return text
    
    def make_html(self) -> pd.DataFrame:
        """
        Генерация HTML разметки для глав документа с таблицами и изображениями.
        
        Returns:
            pd.DataFrame: DataFrame с сгенерированным HTML содержимым
        """
        data_size = self.document.shape[0]
        
        html_template = self._get_html_template()
        
        for i in range(data_size):
            if self.document.iloc[i]["CleanText"] or self.document.iloc[i]["Text"]:
                header = self.document.iloc[i]["Header"]
                text = (self.document.iloc[i]["CleanText"] 
                       if self.is_pdf and self.document.iloc[i]["CleanText"] 
                       else self.document.iloc[i]["Text"])
                tables = self.document.iloc[i]["Tables"]
                images = self.document.iloc[i]["Images"]
                
                # Генерация HTML содержимого
                content_html = self._generate_chapter_html(header, text, tables, images)
                
                # Создание финального HTML
                final_html = html_template.format(
                    title=header,
                    content=content_html
                )
                
                self.document.loc[i, 'HTML'] = final_html
                self.logger.info(f"Сгенерирован HTML для главы {i+1}: {header}")
        
        return self.document
    
    def _get_html_template(self) -> str:
        """Получение HTML шаблона для генерации документа."""
        return '''<!DOCTYPE html>
<html lang="ru">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 40px;
            color: #333;
            font-size: 16px;
            background-color: #f5f5f5;
        }}
        .container {{
            max-width: 900px;
            margin: 0 auto;
            background-color: white;
            padding: 40px;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #2c3e50;
            font-size: 32px;
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
            margin-bottom: 30px;
        }}
        h2 {{
            color: #2980b9;
            font-size: 24px;
            margin-top: 30px;
            margin-bottom: 15px;
        }}
        p {{
            margin-bottom: 15px;
            text-align: justify;
        }}
        table {{
            border-collapse: collapse;
            width: 100%;
            margin: 30px 0;
            background-color: white;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }}
        th, td {{
            border: 1px solid #ddd;
            padding: 12px;
            text-align: left;
        }}
        th {{
            background-color: #3498db;
            color: white;
            font-weight: bold;
        }}
        tr:nth-child(even) {{
            background-color: #f2f2f2;
        }}
        .figure {{
            margin: 30px 0;
            text-align: center;
        }}
        .figure img {{
            max-width: 100%;
            height: auto;
            border: 1px solid #ddd;
            border-radius: 4px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }}
        .figure-caption {{
            margin-top: 10px;
            font-style: italic;
            color: #666;
            font-size: 14px;
        }}
        .highlight {{
            background-color: #f9e79f;
            font-weight: bold;
            padding: 2px 4px;
            border-radius: 3px;
        }}
    </style>
</head>
<body>
    <div class="container">
        {content}
    </div>
</body>
</html>'''
    
    def _generate_chapter_html(self, header: str, text: str, 
                              tables: List[Dict], images: List[Dict]) -> str:
        """Генерация HTML содержимого для одной главы."""
        content_html = f"<h1>{header}</h1>\n"
        
        paragraphs = text.split('\n\n')
        was_table = False
        
        for paragraph in paragraphs:
            if paragraph.strip():
                # Обработка маркеров таблиц
                if paragraph.startswith('\n[Таблица') or paragraph.startswith(' [Таблица') or paragraph.startswith('[Таблица'):
                    table_match = re.search(r'\[Таблица (\d+)\]', paragraph)
                    if table_match and tables:
                        table_id = int(table_match.group(1))
                        was_table = True
                        for table in tables:
                            if table.get('id') == table_id:
                                content_html += table.get('html', '')
                                break
                
                elif was_table:
                    was_table = False
                    continue
                
                # Обработка маркеров изображений
                elif paragraph.startswith('\n[Изображение'):
                    img_match = re.search(r'\[Изображение (\d+):', paragraph)
                    if img_match and images:
                        img_id = int(img_match.group(1))
                        for img in images:
                            if img.get('id') == img_id:
                                content_html += self._generate_image_html(img)
                                break
                
                # Обычный параграф
                else:
                    content_html += f"<p>{paragraph}</p>\n"
        
        return content_html
    
    def _generate_image_html(self, img: Dict) -> str:
        """Генерация HTML для элемента изображения."""
        html = '<div class="figure">\n'
        if img.get('path'):
            html += f'<img src="{img["path"]}" alt="{img.get("caption", "")}">\n'
        if img.get('caption'):
            html += f'<div class="figure-caption">{img["caption"]}</div>\n'
        html += '</div>\n'
        return html
    
    def get_document(self) -> pd.DataFrame:
        """Получение DataFrame документа."""
        return self.document
    
    def get_images(self) -> List[Dict]:
        """Получение списка извлеченных изображений."""
        return self.images_data
    
    def get_tables(self) -> List[Dict]:
        """Получение списка извлеченных таблиц."""
        return self.tables_data
    
    def get_parceby_level(self) -> int:
        """Получение уровня разделения на главы."""
        return self.document_config.parceby_level
    
    def set_parceby_level(self, level: int) -> None:
        """Установка уровня разделения на главы."""
        self.document_config.parceby_level = level
    
    def load(self, file_name: str) -> None:
        """Загрузка документа из CSV файла."""
        try:
            self.document = pd.read_csv(file_name, sep=";")
            self.logger.info(f"Документ загружен из {file_name}")
        except Exception as e:
            self.logger.error(f"Не удалось загрузить документ из {file_name}: {e}")
            raise
    
    def save(self, file_name: Optional[Path] = None) -> None:
        """Сохранение документа в CSV файл."""
        try:
            if not file_name:
                file_name = self.output_dir / Path(f"{self.file_name}.csv")
            self.document.to_csv(file_name, sep=";", index=False)
            self.logger.info(f"Документ сохранен в {file_name}")
        except Exception as e:
            self.logger.error(f"Не удалось сохранить документ в {file_name}: {e}")
            raise
    def clean_documents(self):
        self.document = pd.DataFrame(columns=['Header', 'Text', 'CleanText', 'HTML', 'Tables', 'Images'])
        self.images_data: List[Dict] = []
        self.tables_data: List[Dict] = []

    def _generate_ids_sync(self, count: int) -> List[str]:
        """Генерация уникальных ID с сервера."""
        try:
            res = self.ids_service.ids_generator.generate_id_list(count)
            # host = f"{self.server_url}/generate_id"
            # message = {"count": count}
            # response = httpx.post(
            #     url=host,
            #     json=message,
            #     timeout=45.0
            # )
            # response.raise_for_status()
            # res = response.json()
            return res # res.get("ids", [])
        except Exception as e:
            self.logger.warning(f"Не удалось сгенерировать ID: {e}, используется резервный вариант. res: {res}")
            return [f"id-{i}" for i in range(count)]
    
    def create_html_files(self, path_html_pages: str) -> None:
        """Сохранение сгенерированной HTML разметки в файлы с уникальными именами."""
        html_id_list = self._generate_ids_sync(len(self.document))
        self.logger.info(f"Сгенерировано {len(html_id_list)} уникальных ID")
        
        if not html_id_list:
            return
        
        os.makedirs(path_html_pages, exist_ok=True)
        self.document["HTML_name"] = ""
        
        i = 0
        for j in range(self.document.shape[0]):
            if (self.document["HTML"][j] and 
                self.document["HTML"][j] != "-" and 
                len(self.document["HTML"][j])):
                
                file_path = os.path.join(path_html_pages, f"{html_id_list[i]}.html")
                self.document.loc[j, 'HTML_name'] = f"{html_id_list[i]}.html"
                
                try:
                    with codecs.open(file_path, "w", "utf-8-sig") as file:
                        html_content = self.document["HTML"][j]
                        if html_content[:3] == "```":
                            file.write(html_content[7:-3])
                        else:
                            file.write(html_content)
                    
                    self.logger.info(f"Сохранен HTML файл {i+1}: {self.document.iloc[j]['HTML_name']}")
                    i += 1
                except Exception as e:
                    self.logger.error(f"Не удалось сохранить HTML файл {i+1}: {e}")
            else:
                self.logger.error(f"Отсутствует HTML {j}")
        self.save(self.output_dir / Path(f"{self.file_name}.csv"))
    
    def create_txt_files(self, path_txt_pages: str) -> None:
        """Сохранение извлеченного текста в файлы, названные по заголовкам."""
        os.makedirs(path_txt_pages, exist_ok=True)
        
        for i in range(self.document.shape[0]):
            if (self.document["Text"][i] and 
                self.document["Text"][i] != "-" and 
                len(self.document["Text"][i])):
                
                # Очистка имени файла
                file_name = self.document['Header'][i].replace(',', '').replace('/', '').replace('\\', '').replace(':', '')
                if file_name=='':
                    file_name = f"{i}"
                file_path = os.path.join(path_txt_pages, f"{file_name}.txt")
                
                try:
                    with codecs.open(file_path, "w", "utf-8-sig") as file:
                        file.write(self.document["Text"][i])
                    self.logger.info(f"Сохранен текстовый файл {i+1}: {file_name}.txt")
                except Exception as e:
                    self.logger.error(f"Не удалось сохранить текстовый файл {i+1}: {e}")

