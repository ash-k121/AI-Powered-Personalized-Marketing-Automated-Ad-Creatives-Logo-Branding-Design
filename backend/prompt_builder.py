def build_prompt(data):
    brand = data.get("brand_name", "Your Brand")
    category = data.get("category", "tech")
    style = data.get("style", "modern, minimalist")
    color = data.get("colors", "blue and white")
    description = data.get("description", "")

    prompt = f"{style} logo design for a {category} brand called '{brand}', using {color} colors. {description}"
    return prompt
