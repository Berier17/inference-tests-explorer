# -*- coding: utf-8 -*-
"""
Created on Fri Aug 22 13:27:14 2025

@author: aliel
"""
#Importing libraries
from manim import *
import numpy as np



class SamplingDistribution(Scene):
    def construct(self):
        # Title 
        title = Text("Sampling Distribution of Sample Means", font_size=32)
        title.to_edge(UP)
        self.play(Write(title))
        self.wait(1)
        
        # Create axes for population distribution (no numbers to avoid LaTeX)
        axes_pop = Axes(
            x_range=[-4, 4, 1],
            y_range=[0, 0.5, 0.1],
            axis_config={"include_numbers": False},
            x_length=5,
            y_length=3,
            tips=False
        ).shift(LEFT * 3.5 + UP * 0.5)
        
        # Population distribution (normal curve)
        def normal_func(x):
            return (1/np.sqrt(2*np.pi)) * np.exp(-x**2/2)
        
        pop_curve = axes_pop.plot(normal_func, color=BLUE, x_range=[-3.5, 3.5])
        pop_label = Text("Population Distribution", font_size=24, color=BLUE)
        pop_label.next_to(axes_pop, DOWN, buff=0.3)
        
        self.play(Create(axes_pop))
        self.play(Create(pop_curve))
        self.play(Write(pop_label))
        self.wait(1)
        
        # Create axes for sampling distribution (no numbers to avoid LaTeX)
        axes_samp = Axes(
            x_range=[-2, 2, 0.5],
            y_range=[0, 1.5, 0.3],
            axis_config={"include_numbers": False},
            x_length=5,
            y_length=3,
            tips=False
        ).shift(RIGHT * 3.5 + UP * 0.5)
        
        samp_label = Text("Sampling Distribution\nof Sample Means", font_size=20, color=GREEN)
        samp_label.next_to(axes_samp, DOWN, buff=0.3)
        
        self.play(Create(axes_samp))
        self.play(Write(samp_label))
        self.wait(1)
        
        # Generate population data
        np.random.seed(42)  # For reproducible results
        population = np.random.normal(0, 1, 1000)
        
        # Show some population points
        sample_points = np.random.choice(population, 50)
        dots = VGroup()
        for x in sample_points:
            if -3.5 <= x <= 3.5:  # Only show points within visible range
                dot = Dot(axes_pop.c2p(x, normal_func(x) + np.random.uniform(-0.02, 0.02)), 
                         color=YELLOW, radius=0.04)
                dots.add(dot)
        
        self.play(LaggedStart(*[GrowFromCenter(dot) for dot in dots], lag_ratio=0.02))
        self.wait(1)
        
        # Simulate sampling process
        sample_means = []
        bars = VGroup()
        n_samples = 30
        sample_size = 5
        
        # Create bins for histogram
        bin_width = 0.3
        bins = np.arange(-2, 2.1, bin_width)
        bin_counts = np.zeros(len(bins)-1)
        
        for i in range(n_samples):
            # Take a random sample
            sample = np.random.choice(population, sample_size, replace=False)
            mean_val = np.mean(sample)
            sample_means.append(mean_val)
            
            # Find which bin this mean belongs to
            bin_idx = np.digitize(mean_val, bins) - 1
            if 0 <= bin_idx < len(bin_counts):
                bin_counts[bin_idx] += 1
                
                # Create/update bar for this bin
                bar_height = bin_counts[bin_idx] * 0.1
                bin_center = (bins[bin_idx] + bins[bin_idx + 1]) / 2
                
                # Remove old bar if it exists
                old_bars = [bar for bar in bars if abs(bar.get_center()[0] - axes_samp.c2p(bin_center, 0)[0]) < 0.1]
                if old_bars:
                    bars.remove(*old_bars)
                    self.remove(*old_bars)
                
                # Create new bar
                bar = Rectangle(
                    width=bin_width * axes_samp.x_length / (axes_samp.x_range[1] - axes_samp.x_range[0]),
                    height=bar_height * axes_samp.y_length / (axes_samp.y_range[1] - axes_samp.y_range[0]),
                    color=GREEN,
                    fill_opacity=0.7,
                    stroke_width=1
                )
                bar.move_to(axes_samp.c2p(bin_center, bar_height/2))
                bars.add(bar)
                
                self.play(FadeIn(bar), run_time=0.2)
        
        self.wait(1)
        
        # Add theoretical sampling distribution curve
        def sampling_dist_func(x):
            # Theoretical: normal with mean=0, std=population_std/sqrt(sample_size)
            std_error = 1 / np.sqrt(sample_size)  # population std = 1
            return (1/(std_error * np.sqrt(2*np.pi))) * np.exp(-x**2/(2*std_error**2))
        
        theoretical_curve = axes_samp.plot(
            sampling_dist_func, 
            color=RED, 
            x_range=[-2, 2],
            stroke_width=3
        )
        
        theory_label = Text("Theoretical      Distribution", font_size=18, color=RED)
        theory_label.next_to(axes_samp, IN, buff=0.3).shift(UP * 1.5)
        
        self.play(Create(theoretical_curve))
        self.play(Write(theory_label))
        self.wait(1)
        
        # Add statistics
        sample_mean = np.mean(sample_means)
        sample_std = np.std(sample_means)
        theoretical_std = 1 / np.sqrt(sample_size)
        
        stats_text = VGroup(
            Text(f"Sample mean: {sample_mean:.3f}", font_size=20),
            Text(f"Sample std: {sample_std:.3f}", font_size=20),
            Text(f"Theoretical std: {theoretical_std:.3f}", font_size=20),
            Text(f"Sample size: {sample_size}", font_size=20),
            Text(f"Number of samples: {n_samples}", font_size=20)
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.2)
        stats_text.to_edge(DOWN, buff=0.5)
        
        self.play(Write(stats_text))
        self.wait(2)
        
        # Conclusion
        conclusion = Text(
            "Central Limit Theorem: Sample means approach normal distribution!",
            font_size=24,
            color=YELLOW
        ).to_edge(UP, buff=0.1)
        
        self.play(Transform(stats_text, conclusion))
        self.wait(3)



class Perspective4D(ThreeDScene):
    def construct(self):
        # Set up 3D axes
        axes = ThreeDAxes(
            x_range=[-5, 5, 1],
            y_range=[-5, 5, 1],
            z_range=[-3, 3, 1],
            x_length=8,
            y_length=8,
            z_length=4,
            axis_config={'include_numbers': False, 'stroke_width': 2}
        )
        
        # Axis labels
        x_label = Text("X", font_size=24, color=WHITE).move_to(axes.c2p(5.5, 0, 0))
        y_label = Text("Y", font_size=24, color=WHITE).move_to(axes.c2p(0, 5.5, 0))
        
        # Generate 3D data points that create an illusion
        np.random.seed(42)
        n_points = 20
        
        # Create points that look linear from XY view but are scattered in Z
        x_data = np.linspace(-4, 4, n_points) + np.random.normal(0, 0.3, n_points)
        y_data = 0.8 * x_data + np.random.normal(0, 0.5, n_points)  # Linear in XY
        z_data = np.random.normal(0, 1.5, n_points)  # Random Z coordinates
        
        # Create data points
        data_points = VGroup()
        for i in range(n_points):
            point = Sphere(radius=0.12, color=RED, resolution=(8, 6))
            point.move_to(axes.c2p(x_data[i], y_data[i], z_data[i]))
            data_points.add(point)
        
        # Create ONE line that goes through 3D space
        # This line will look "wrong" from 2D view but "right" from 3D view
        x_line = np.linspace(-4, 4, 50)
        y_line = 0.6 * x_line + 0.5  # Slight offset
        z_line = 0.3 * x_line  # The line goes through 3D space!
        
        regression_line_points = [axes.c2p(x_line[i], y_line[i], z_line[i]) for i in range(len(x_line))]
        regression_line = VMobject()
        regression_line.set_points_smoothly(regression_line_points)
        regression_line.set_stroke(color=GREEN, width=6)
        
        # Title and labels
        title = Text("Regression Line Perspective Illusion", 
                    font_size=28, color=BLUE).to_edge(UP)
        
        # Start with 2D view (looking down XY plane)
        self.set_camera_orientation(phi=0 * DEGREES, theta=-90 * DEGREES, distance=12)
        
        # Animation sequence
        self.play(Write(title))
        self.wait(1)
        
        self.play(Create(axes))
        self.play(Write(x_label), Write(y_label))
        self.wait(1)
        
        # Show data points - from 2D they look linear
        self.play(LaggedStart(*[GrowFromCenter(point) for point in data_points], 
                            lag_ratio=0.1))
        self.wait(1)
        
        # Show the line - from 2D it looks wrong!
        wrong_label = Text("This line looks wrong from 2D view!", 
                          font_size=20, color=RED).to_edge(DOWN)
        self.play(Write(wrong_label))
        self.play(Create(regression_line), run_time=2)
        self.wait(2)
        
        # Now rotate to reveal why it's actually correct!
        revelation_text = Text("But from 3D perspective...", 
                             font_size=20, color=BLUE).to_edge(DOWN)
        self.play(Transform(wrong_label, revelation_text))
        
        # Dramatic camera rotation to reveal Z dimension
        self.move_camera(phi=70 * DEGREES, theta=-45 * DEGREES, run_time=4)
        self.wait(2)
        
        # Add Z label now that we can see it
        z_label = Text("Z", font_size=24, color=WHITE).move_to(axes.c2p(0, 0, 3.5))
        z_label.rotate(PI/2, axis=RIGHT)  # Make it face the camera after rotation
        self.play(Write(z_label))
        self.wait(1)
        
        # Show the actual best fit line
        perfect_fit_label = Text("Perfect! The line goes through the 3D data!", 
                               font_size=20, color=GREEN).to_edge(DOWN)
        self.play(Transform(wrong_label, perfect_fit_label))
        self.wait(3)
        
      
        
        self.begin_ambient_camera_rotation(rate=0.2)
        self.wait(8)
        self.stop_ambient_camera_rotation()
        
        # Add explanation
        explanation = Text("The same line can look wrong from one angle\nbut perfect from another perspective!", 
                         font_size=21, color=BLUE_D)
        explanation.move_to([-11,-11,-11])
        self.add_fixed_orientation_mobjects(explanation)
        self.play(Write(explanation))
        self.wait(2)
        
        # Final ambient rotation to show the perspective illusion
        final_text = Text("Perspective is  everything!", 
                      font_size=21, color=GREEN_E)
        final_text.to_edge(RIGHT)
        self.add_fixed_orientation_mobjects(final_text)
        
        
        self.wait(2)
