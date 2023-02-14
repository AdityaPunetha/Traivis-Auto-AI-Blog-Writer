from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM

_HEADINGS_GENERATION_PROMPT = """Generate the headings for the sections of a blog post for the following title: Staying Productive While Working from Home.
1. Creating a Dedicated Workspace
2. Making a Schedule and Sticking to It
3. Taking Breaks and Moving Around
4. Communicating with Colleagues

Generate the headings for the sections of a blog post for the following title: How to Plan a Successful Road Trip
1. Choose a destination
2. Choose some of the best hotels
3. Get a flight out of it
4. Pack Smartly
5. Staying Safe on the Road

Generate the headings for the sections of a blog post for the following title: {topic}
1."""

_SECTION_GENERATION_PROMPT = """Blog title: Generate the headings for the sections of a blog post for the following title: Staying Productive While Working from Home.:
Section title: Making a Schedule and Sticking to It:
<|startoftext|>
Working from home can be a great way to increase productivity and eliminate the distractions of a traditional office. However, it can also be easy to fall into the trap of procrastination and disorganization. One of the best ways to stay on track when working from home is to make a schedule and stick to it.

Start by creating a daily schedule that includes time for work, breaks, and any other activities you need to accomplish. Make sure to set specific times for each task and try to stick as closely to the schedule as possible. This will help you stay focused and on task throughout the day.

One key to sticking to your schedule is to set realistic goals for yourself. Don't try to accomplish too much in one day, as this can lead to burnout and frustration. Instead, focus on completing a few important tasks each day and build up to more as you become more comfortable with working from home.

Another important aspect of staying productive while working from home is to create a dedicated workspace. This can be a separate room or just a designated area in your home where you can work without interruption. Make sure your workspace is comfortable and free from distractions, and try to use it exclusively for work-related tasks.

Finally, it's important to take breaks throughout the day. Working non-stop can quickly lead to burnout, so make sure to schedule in some time for rest and relaxation. You can take a short walk, do some stretching exercises, or simply take a few minutes to clear your mind.

Overall, making a schedule and sticking to it is crucial for staying productive while working from home. By setting realistic goals, creating a dedicated workspace, and taking regular breaks, you can stay focused and on track throughout the day.
<|endoftext|>


Blog title: How to Plan a Successful Road Trip
Section title: Choose a destination
<|startoftext|>
When planning a road trip, choosing a destination is a crucial step that sets the tone for the entire trip. It's important to consider your interests, budget and the time of year to choose the perfect destination. Start by thinking about the activities and experiences you want to have on your trip, and look for destinations that align with those interests. For example, if you're an outdoor enthusiast, look for destinations with plenty of hiking trails and natural beauty.

Next, set a budget and choose a destination that fits within that budget. This will help you narrow down your options and ensure that you don't overspend on your trip. Research different destinations by looking at photos, reading reviews, and checking out online travel guides to get a better idea of what to expect.

When choosing a destination, it's also important to consider the time of year. If you're planning to take a trip in the summer, for example, choose a destination with plenty of outdoor activities and warm weather. And don't forget to be open-minded. Sometimes, the most unexpected destinations end up being the most memorable. Consider places you've never been to before or don't have as much fame as others.

In summary, when planning a road trip, choosing a destination that aligns with your interests, budget and schedule and that will help make your road trip a success. Take your time, do your research, and be open-minded when choosing your destination. And most importantly, have fun!
<|endoftext|>


Blog title: {topic}
Section title: {section}
<|startoftext|>
"""


class Generator:
    def __init__(self, model_name="EleutherAI/gpt-neo-1.3B"):
        self.model = pipeline("text-generation", model=model_name, device=0)
        self._section_generation_prompt = _HEADINGS_GENERATION_PROMPT
        self._section_expansion_prompt = _SECTION_GENERATION_PROMPT

    def generate_blog_headings(self, topic):
        prompt = self._section_generation_prompt.format(topic=topic)
        output = self.model(prompt, max_length=256, do_sample=True, temperature=0.7)[0][
            "generated_text"
        ]
        return self.parse_blog_headings(output)

    def parse_blog_headings(self, sections):
        parsed = sections.split("\n\n")[2]
        sections_list = [x[2:] for x in parsed.split("\n")]
        return sections_list[1:]

    def generate_blog_section(self, title, section):
        prompt = self._section_expansion_prompt.format(topic=title, section=section)
        output = self.model(
            prompt, max_new_tokens=512, do_sample=True, temperature=0.7
        )[0]["generated_text"]
        return self.parse_blog_section(output)

    def parse_blog_section(self, section):
        parsed = section.split("<|startoftext|>")[3]
        return parsed
