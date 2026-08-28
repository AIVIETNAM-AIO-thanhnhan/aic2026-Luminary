"""Hand-translated English visual descriptions for the 25 exam queries.

Gemini's free-tier quota is exhausted (429 RESOURCE_EXHAUSTED), so query
expansion can't run through the LLM path right now. These are manual
translations of the Vietnamese query text, written with full context of each
question, used in place of the Gemini decomposition step.
"""

from __future__ import annotations

VISUAL_EN: dict[str, list[str]] = {
    "query-p1-1-kis": [
        "A large group of more than five people standing in rows doing group "
        "exercise, bending down to touch their toes with both hands",
        "Among a group of people exercising, only one wears glasses and three wear red hats",
        # A blunt, keyword-dense phrasing alongside the two prose variants above -
        # CLIP/SigLIP text towers sometimes match a plain fact list better than a
        # flowing sentence, especially for multiple simultaneous counts.
        "6 people exercising in a line, 1 person wearing glasses, 3 people wearing red hats",
    ],
    "query-p1-2-kis": [
        "A map with an irrigation dam icon appearing four times on it",
        "An aerial drone view of a large dam",
        "A close-up of a dam wall in heavy rain",
    ],
    "query-p1-3-qa": [
        "A fish being weighed on a scale",
        "A person holding a fish by its tail",
        "A scale displaying a number with a fish on it",
    ],
    "query-p1-4-kis": [
        "A pride of lions resting and climbing on wooden platforms in a zoo enclosure",
        "A London Zoo information sign in front of an animal enclosure",
        "Two zookeepers in green shirts weighing an animal and recording data",
    ],
    "query-p1-5-kis": [
        "Green peas being added to squid stir-fried in a wok",
        "Sliced onions and red chili peppers on a plate next to a stir-fry pan",
        "Slow motion shot of a chef shaking a wok over open flame",
    ],
    "query-p1-6-kis": [
        "A man in a dark blue suit and tie sitting in a large chair holding a "
        "large rough gemstone close to his face",
        "A woman in black office attire with a pink headscarf standing and smiling",
        "Aerial view of a large open-pit gemstone mine with deep terraced excavation",
    ],
    "query-p1-7-kis": [
        "Star-shaped carrot slices boiling in a pot of water in a metal strainer, "
        "stirred with wooden chopsticks",
        "A plated dish of boiled and fried vegetables: okra, cauliflower, "
        "star-shaped carrots, zucchini, a pink dipping sauce in the middle, "
        "light pink chopsticks on the right",
    ],
    "query-p1-8-kis": [
        "A chef placing bar-shaped ingredients and flower-shaped cut pieces onto "
        "a plate of food being steamed in a pot",
        "A chef using chopsticks to arrange ingredients around food already on a plate",
        "A chef using a spoon to scoop a soft ingredient from a glass bowl onto the "
        "center of a plate",
    ],
    "query-p1-9-qa": [
        "Cars driving through deep flood water one after another: a yellow car, a "
        "red car, and a black car about to cross under a bridge",
        "A number sign on the left side of a bridge over flooded water",
    ],
    "query-p1-10-kis": [
        "A bunch of grapes being cut from a vine with black scissors",
        "A blue string tied around the stem of a grape bunch before it is cut",
    ],
    "query-p1-11-kis": [
        "A slow motion shot at the finish line of a bicycle race, low camera angle "
        "at road level",
        "Cyclists crossing the finish line: first in yellow jersey and black shorts, "
        "second in blue jersey and black shorts, third in blue jersey and red shorts",
    ],
    "query-p1-12-kis": [
        "Four motorbike taxi drivers at a gas station, three standing waiting and "
        "one riding a motorbike from left to right",
        "A person closing the gas cap of a motorbike",
        "A price board showing mazut fuel oil price at a gas station",
    ],
    "query-p1-13-kis": [
        "A person standing in water at night shining a flashlight",
        "A person pulling a fishing net out of water at sunrise dawn",
        "A film crew with a camera approaching a fisherman at dawn",
    ],
    "query-p1-14-kis": [
        "A chef placing bar-shaped ingredients and flower-shaped cut pieces onto "
        "a plate of food being steamed in a pot",
        "A chef using chopsticks to arrange ingredients around food already on a plate",
        "A chef using a spoon to scoop a soft ingredient from a glass bowl onto the "
        "center of a plate",
    ],
    "query-p1-15-qa": [
        "A world or regional earthquake distribution map with a legend on the left "
        "side showing colored symbols for different earthquake magnitude levels",
        "An earthquake map with colored dots marking epicenter locations",
    ],
    "query-p1-16-trake": [
        "A close-up of a white lion dance head with a red nose next to a white "
        "flag with red border",
        "A lion dance performance with dragons, drums, and poles",
    ],
    "query-p1-17-qa": [
        "A mountain pass road completely blocked by a landslide with dirt and rocks",
        "A close-up of a road boundary marker post with a red top, mostly buried "
        "in dirt and rocks",
        "A person riding a motorbike struggling through muddy terrain with a "
        "green object hanging on the bike",
    ],
    "query-p1-18-kis": [
        "A chef ladling broth and ingredients like chicken, carrot, lemongrass, "
        "and wood ear mushroom into a bowl of rice noodles, finishing with cilantro on top",
        "A bowl of Vietnamese noodle soup next to a small dipping sauce bowl with "
        "two pieces of chili, camera zooming out",
    ],
    "query-p1-19-kis": [
        "A lion dance costume operated by two people standing and spinning on "
        "top of a pole",
        "A lion dance performer jumping across poles and grabbing a pumpkin "
        "decorated with a yellow flower with its mouth",
    ],
    "query-p1-20-kis": [
        "Three people walking down a slope in the rain, two holding umbrellas, "
        "one wearing a raincoat with a bear print on the back",
        "A group of people walking together on a dirt path toward a house next "
        "to a pond",
    ],
    "query-p1-21-kis": [
        "Peeled cooked shrimp arranged on a plate, with a chef placing three "
        "baguettes on a table in the background",
        "Chefs decorating and cooking food in a kitchen",
        "Shrimp cut in half and grilled on a stove",
    ],
    "query-p1-22-kis": [
        "A woman wearing a pink traditional Vietnamese ao dai dress and glasses "
        "teaching an English grammar lesson about the verb remember",
    ],
    "query-p1-23-kis": [
        "A male teacher wearing a white shirt and dark tie in front of a dark "
        "blue patterned background",
        "An educational slide with a white background, pink and purple border, "
        "and a blue header bar with a globe icon",
        "A three tier diagram connected by teal arrows: two boxes inside an "
        "orange box on top, one large dark blue box in the middle, two boxes "
        "inside a green box at the bottom",
    ],
    "query-p1-24-kis": [
        "A single panning camera shot from left to right showing handmade water "
        "hyacinth woven products: a handbag, a flower pot, a tea set, and another handbag",
        "A woman picking up a teacup from a tea set and holding it gently while "
        "another woman talks to her",
    ],
    "query-p1-25-kis": [
        "Two students wearing white shirts, blue pants, and red scarves acting "
        "as MCs on a school stage",
        "A school stage with a red drum kit and a piano in the background",
    ],
}

# TRAKE event-by-event English descriptions, in order, for query-p1-16.
TRAKE_EVENTS: dict[str, list[str]] = {
    "query-p1-16-trake": [
        "Two golden dragons fully visible, spinning/twirling together",
        "A lion dance performer completing a spin on top of poles, feet landing "
        "back on the poles right after the spin",
        "A mallet or stick striking a gong",
    ],
}
