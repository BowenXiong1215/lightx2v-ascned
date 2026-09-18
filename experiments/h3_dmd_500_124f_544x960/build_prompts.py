"""Build a diverse, text-only audiovisual prompt set for H3 DMD."""

from pathlib import Path


# Each entry pairs a visible scene/action with an identifiable natural sound.
# The three camera variants make 300 training prompts from 100 distinct events.
TRAIN_EVENTS = """
a city tram rounds a corner on wet rails|the tram bell rings and metal wheels squeal softly
a bicycle courier rides over cobblestones|the bicycle chain rattles and tires click over the stones
a bus pulls away from a curb at dusk|the engine hum rises and the air brakes hiss
a subway train arrives at an underground platform|the train rumbles, brakes squeal, and doors chime
a freight train crosses a rural level crossing|the crossing bell rings over rhythmic wheel clatter
a small ferry approaches a harbor pier|water slaps the hull and a low horn sounds
a fishing boat motors across a calm bay|the outboard motor drones over small splashes
a helicopter passes above an empty parking lot|rotor blades pulse and fade into the distance
a delivery van reverses into a loading bay|a regular reversing beep sounds above the engine
a fire engine drives along a city avenue|a two-tone siren alternates clearly as it passes
a street sweeper moves along a quiet road|its brushes scrape and its motor whirs
a skateboarder rolls down a smooth plaza ramp|wheels rumble and the board taps the concrete
a pedestrian crosses a wooden footbridge|measured footsteps thump on the planks
a car drives slowly through a gravel driveway|tires crunch on gravel and the engine idles
a motorcycle passes through a tunnel|the engine revs and briefly echoes
a sailboat moves beside a marina|rigging clinks and water laps against the boat
a tugboat pushes a barge along a river|a deep engine throb mixes with river splashes
a cable car glides above a mountain valley|the cable mechanism hums and wind brushes past
a row of traffic lights changes beside an intersection|distant cars pass and a pedestrian signal clicks
a bicycle mechanic spins a wheel in a street workshop|the freewheel ticks and tools clink
a barista prepares espresso in a quiet cafe|the grinder buzzes and the espresso machine hisses
a cook chops vegetables on a wooden board|a knife makes quick rhythmic taps against the board
a kettle reaches a boil on a kitchen stove|water bubbles and the kettle whistles
a chef stirs soup in a metal pot|the spoon scrapes the pot and liquid simmers
a baker kneads dough on a flour-covered counter|soft dough slaps the counter and trays clatter nearby
a toaster pops beside a breakfast table|a small mechanical click is followed by the toast popping up
a dishwasher opens after a wash cycle|water drips and dishes clink as the rack slides out
a person washes a glass in a kitchen sink|running water splashes and glass taps ceramic
a washing machine spins in a laundry room|the drum turns with a low repeating rumble
a door opens into a quiet hallway|the latch clicks, hinges creak, and footsteps enter
a ceiling fan rotates above an empty room|the fan makes a steady low whirr
a grandfather clock stands in a living room|a distinct tick-tock repeats and the clock chimes
a child stacks wooden blocks on a rug|blocks knock together with short hollow clicks
a person types quickly at a desk|distinct mechanical keyboard clicks form a rapid rhythm
a printer produces a page in an office|rollers whir and the paper slides into the tray
a sewing machine stitches fabric at a table|the needle makes fast even clicks
a carpenter saws a plank in a workshop|the saw rasps through wood and sawdust falls
a carpenter hammers nails into a board|each hammer strike lands with a sharp wooden knock
a drill presses into a metal plate|the drill motor whines and the metal vibrates
a mechanic tightens a bolt under a car|a ratchet clicks repeatedly and a wrench clinks
a lathe shapes a small wooden bowl|the machine hums and a tool scrapes wood
a pottery wheel spins as hands shape clay|the motor hums and wet clay rubs against fingertips
a blacksmith strikes heated iron on an anvil|bright hammer clangs ring through the workshop
a garage door rises on its tracks|the motor drones and metal rollers rattle
a forklift lifts a pallet inside a warehouse|the hydraulic lift whines and its warning beep sounds
a construction crane turns above a building site|the motor groans and distant tools strike metal
a metal gate rolls shut at a factory|the wheels scrape and the gate lands with a heavy clang
a glassblower rotates a glowing piece of glass|the furnace roars steadily in the background
a farmer drives a tractor across a field|the diesel engine rumbles and tires press through soil
a lawn mower crosses a small garden|the mower buzzes steadily and grass rustles
a windy forest clearing with tall pine trees|wind moves through needles and branches creak
a stream runs over smooth stones in a forest|clear flowing water gurgles over the rocks
a waterfall drops into a rocky pool|falling water roars continuously and echoes
ocean waves roll onto a sandy beach|each wave breaks, foams, and retreats across sand
small waves hit a wooden lakeside dock|water slaps the posts and the dock creaks
heavy rain falls onto a greenhouse roof|individual raindrops patter on glass above a low rain wash
rain begins on a dry city sidewalk|scattered drops become steady pattering on pavement
thunderclouds gather above an open field|a distant thunder roll follows the wind
snow falls around a cabin in a pine forest|wind softly whistles around the eaves
dry leaves blow across an autumn path|leaves scrape the ground in irregular gusts
bamboo stalks sway beside a garden pond|bamboo leaves rustle and stems tap lightly
a campfire burns beside a quiet campsite|wood crackles and an occasional ember pops
a person walks through fresh snow|boots make crisp crunching steps
a person walks through shallow water at the shore|footsteps splash and water drains away
a mountain river runs beneath a stone bridge|fast water rushes and echoes under the bridge
a desert wind blows across a field of dunes|a steady low wind brushes the sand
a small fountain sprays water in a courtyard|water splashes in a repeating pattern
hail strikes a metal garden shed|small hard impacts rattle quickly on the roof
ice cubes fall into a glass in a kitchen|the cubes clink sharply and liquid splashes
a cat watches birds from an open window|the cat meows once while birds chirp outside
a dog runs across a grassy yard|the dog barks and paws thud softly on grass
a puppy drinks from a water bowl|small lapping sounds and the bowl moves slightly
a horse trots along a dirt track|hooves beat a steady rhythm on packed earth
a cow walks through a quiet pasture|a low moo sounds above grass and wind
a flock of sheep moves past a farm gate|soft bleats and hoofsteps mix with a gate creak
ducks swim across a village pond|ducks quack and paddle through water
geese cross a farmyard in a group|loud honks alternate with quick footsteps
a rooster stands on a wooden fence at dawn|a clear rooster crow carries over the yard
a songbird perches on a garden branch|short melodic chirps repeat over quiet ambience
seagulls circle above a harbor|sharp gull calls and distant waves are audible
a woodpecker taps a tree trunk|fast dry tapping repeats in short bursts
frogs sit beside a pond at dusk|several frogs croak in overlapping rhythms
bees move between flowers in a sunny garden|close buzzing rises and falls
a dolphin surfaces beside a boat|water splashes and the animal makes brief squeaks
a crowd applauds a performer on a small stage|hands clap in a broad rhythmic wash
a person claps a short rhythm in a quiet room|individual hand claps are sharp and evenly spaced
a singer rehearses a sustained note in an empty studio|one clear human voice echoes slightly
two friends greet each other outside a cafe|brief natural conversation and soft footsteps are heard
a teacher writes on a classroom chalkboard|chalk scratches and students murmur quietly
a basketball player dribbles across an indoor court|the ball bounces with a hollow echo and shoes squeak
a tennis player serves on an outdoor court|the racket strikes the ball and shoes scrape the court
a soccer player kicks a ball toward a goal|a solid ball impact is followed by running footsteps
a swimmer dives into a pool|a splash breaks the quiet and water ripples
a runner jogs along a park path|breathing and regular footfalls stay in sync
a drummer plays a simple beat on one snare drum|separate snare hits form a steady pattern
a violinist plays a short melody in a practice room|bowed violin notes are clear and slightly resonant
a pianist plays a slow melody at an upright piano|individual piano notes ring and decay
a guitarist strums an acoustic guitar in a living room|bright strings sound in a steady rhythm
a street musician plays a saxophone on a quiet corner|a short saxophone phrase carries over distant traffic
a small brass band marches across a town square|drums keep time beneath bright brass notes
""".strip()

VALIDATION_EVENTS = """
a streetcar waits at a snowy station|its door chime sounds while snow crunches under nearby footsteps
a bicycle bell rings beside a riverside path|a bright bell rings twice over distant river water
a small propeller plane taxis across an airfield|the propeller buzzes and the engine rises in pitch
a harbor buoy moves in choppy water|a metal bell sounds between wave splashes
a person grinds coffee beans in a home kitchen|the grinder buzzes in a short burst
a microwave finishes heating a meal|three clear beeps sound as the door opens
a pencil sharpener turns at a school desk|the sharpener scrapes and stops with a click
a locksmith tests a new key in a door|metal keys jingle and the lock clicks
a mason spreads mortar along a brick wall|the trowel scrapes and taps brick
a welder works behind a safety screen|brief electric crackles sound above a workshop hum
a gardener uses pruning shears on a hedge|the shears snip repeatedly and leaves rustle
a narrow creek runs through a snowy ravine|water trickles beneath ice and wind moves overhead
a tree branch creaks during a storm|wind surges and wood creaks above falling rain
pebbles roll down a steep gravel slope|stones tumble and click against one another
a seal rests on rocks beside the sea|a rough seal bark sounds over distant surf
a donkey walks past a wooden fence|hooves thud and the animal brays
an owl sits in a moonlit tree|a low owl hoot repeats in the quiet night
a woodpecker flies between trees and starts tapping|a brief wing flutter gives way to rapid tapping
a child blows soap bubbles in a sunny yard|a soft breath is followed by distant laughter
a table tennis rally continues in a recreation hall|fast sharp paddle hits alternate across the table
a skateboard lands a small jump on concrete|wheels rattle, then the board lands with a loud crack
a handbell player rings two different bells|two distinct bell pitches alternate cleanly
a flutist practices beside an open window|clear flute notes mix with faint outdoor birds
a crowd counts down before a community race|several voices count together and cheer at the start
""".strip()

CAMERAS = (
    "A continuous five-second locked camera view.",
    "A continuous five-second slow lateral camera move.",
    "A continuous five-second natural handheld view.",
)


def parse_events(block):
    events = []
    for line in block.splitlines():
        scene, sound = line.split("|", 1)
        events.append((scene.strip(), sound.strip()))
    return events


def prompt(scene, sound, camera):
    return f"{camera} In the scene, {scene}. Natural synchronized sound: {sound}. Clear foreground action and realistic background ambience."


def main():
    root = Path(__file__).resolve().parent
    train_events = parse_events(TRAIN_EVENTS)
    val_events = parse_events(VALIDATION_EVENTS)
    train = [prompt(scene, sound, camera) for scene, sound in train_events for camera in CAMERAS]
    val = [prompt(scene, sound, CAMERAS[0]) for scene, sound in val_events]
    if len(train) != len(set(train)) or len(val) != len(set(val)):
        raise RuntimeError("Duplicate prompts found")
    (root / "h3_train_prompts.txt").write_text("\n".join(train) + "\n")
    (root / "h3_val_prompts.txt").write_text("\n".join(val) + "\n")
    print(f"train={len(train)} val={len(val)}")


if __name__ == "__main__":
    main()
